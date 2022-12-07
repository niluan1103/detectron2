try:
    import detectron2
except:
    import os
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')
import numpy as np
import cv2
import random
import glob
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

# Some basic setup:
# Setup detectron2 logger
setup_logger()

# Inference with Detectron2 saved weight
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = "datasets/Wrist-Junk-8/detectron2_wrist_junk_model_final_300.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

def wrist_detect(image_path):
    print(image_path)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    #print(outputs['instances'].get_fields())
    print('classes:',outputs['instances'].pred_classes.tolist())
    print('scores:', outputs['instances'].scores.tolist())
    print('pred_boxes:', (outputs['instances'].pred_boxes.tensor.tolist()))
    v = Visualizer(im[:, :, ::-1],metadata=None,scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    visualized_img = out.get_image()[:, :, ::-1];
    cv2.imwrite('visualized_img.jpg',visualized_img)
    cv2.imshow('visualized_img.jpg', visualized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def for_streamlit(im):
    outputs = predictor(im)
    #print(outputs['instances'].get_fields())
    #print('classes:',outputs['instances'].pred_classes.tolist())
    #print('scores:', outputs['instances'].scores.tolist())
    #print('pred_boxes:', (outputs['instances'].pred_boxes.tensor.tolist()))
    v = Visualizer(im[:, :, ::-1],metadata=None,scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    visualized_img = out.get_image()[:, :, ::-1];
    cv2.imwrite('visualized_img.jpg',visualized_img)
    return visualized_img
    #cv2.imshow('visualized_img.jpg', visualized_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    wrist_detect('test_img.jpg')