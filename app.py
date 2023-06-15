import os
from scipy.spatial import distance
import streamlit as st
from PIL import Image
import torch

import matplotlib.pyplot as plt
from skimage import io

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


def load_model(weights_file, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = DefaultPredictor(cfg)
    return predictor


def detect_damage_part(damage_dict, parts_dict):
    """
    Returns the most plausible damaged part for the list of damages by checking the distance 
    between centers of damage polygons and parts polygons.

    Parameters:
     - damage_dict: dict
                    Dictionary that maps damages to damage polygon centers.
     - parts_dict: dict
                    Dictionary that maps part labels to parts polygon centers.

    Returns:
    - damaged_parts: list
                    The list of damaged part names.
    """
    try:
        max_distance = 10e9
        assert len(damage_dict) > 0, "AssertError: damage_dict should have at least one damage"
        assert len(parts_dict) > 0, "AssertError: parts_dict should have at least one part"
        max_distance_dict = dict(zip(damage_dict.keys(), [max_distance] * len(damage_dict)))
        part_name = dict(zip(damage_dict.keys(), [''] * len(damage_dict)))

        for y in parts_dict.keys():
            for x in damage_dict.keys():
                dis = distance.euclidean(damage_dict[x], parts_dict[y])
                if dis < max_distance_dict[x]:
                    part_name[x] = y.rsplit('_', 1)[0]

        return list(set(part_name.values()))
    except Exception as e:
        print(e)


def main():
    st.title("Car Damage Detection")
    st.write("Upload an image to detect car damages and damaged parts.")

    # Load pre-trained models
    damage_model_weights = "damage_segmentation_model.pth"
    part_model_weights = "parts_weights.pth"

    damage_predictor = load_model(damage_model_weights, num_classes=2)
    part_predictor = load_model(part_model_weights, num_classes=6)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_array = io.imread(uploaded_image)

        # Damage inference
        damage_outputs = damage_predictor(image_array)
        damage_v = Visualizer(
            image_array[:, :, ::-1],
            metadata=MetadataCatalog.get("car_dataset_val"),
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW
        )
        damage_out = damage_v.draw_instance_predictions(damage_outputs["instances"].to("cpu"))

        # Part inference
        parts_outputs = part_predictor(image_array)
        parts_v = Visualizer(
            image_array[:, :, ::-1],
            metadata=MetadataCatalog.get("car_mul_dataset_val"),
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW
        )
        parts_out = parts_v.draw_instance_predictions(parts_outputs["instances"].to("cpu"))

        # Display images with predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
        ax1.imshow(damage_out.get_image()[:, :, ::-1])
        ax2.imshow(parts_out.get_image()[:, :, ::-1])
        st.pyplot(fig)

        damage_prediction_classes = [
            "damage_" + str(indx) for indx in damage_outputs["instances"].pred_classes.tolist()
        ]
        damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
        damage_dict = dict(zip(damage_prediction_classes, damage_polygon_centers))

        parts_prediction_classes = [
            "part_" + str(indx) for indx in parts_outputs["instances"].pred_classes.tolist()
        ]
        parts_polygon_centers = parts_outputs["instances"].pred_boxes.get_centers().tolist()

        # Remove centers which lie beyond 800 units
        parts_polygon_centers_filtered = list(filter(lambda x: x[0] < 800 and x[1] < 800, parts_polygon_centers))
        parts_dict = dict(zip(parts_prediction_classes, parts_polygon_centers_filtered))

        damaged_parts = detect_damage_part(damage_dict, parts_dict)
        st.write("Damaged Parts:", damaged_parts)


if __name__ == "__main__":
    main()
