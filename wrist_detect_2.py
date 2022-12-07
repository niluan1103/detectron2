import detectron2
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
# Register Custom Dataset
register_coco_instances("my_dataset_train", {}, "datasets/Wrist-Junk-8/train/_annotations.coco.json", "datasets/Wrist-Junk-8/train")
register_coco_instances("my_dataset_val", {}, "datasets/Wrist-Junk-8/valid/_annotations.coco.json", "datasets/Wrist-Junk-8/valid")
register_coco_instances("my_dataset_test", {}, "datasets/Wrist-Junk-8/test/_annotations.coco.json", "datasets/Wrist-Junk-8/test")


# Inference with Detectron2 saved weight
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = "datasets/Wrist-Junk-8/detectron2_wrist_junk_model_final_300.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

#cfg.DATASETS.TEST = ("my_dataset_test", )
#test_metadata = MetadataCatalog.get("my_dataset_test")

for imageName in random.sample(glob.glob('datasets/Wrist-Junk-8/test/*jpg'),1):
  print(imageName)
  im = cv2.imread(imageName)
  outputs = predictor(im)
  #print(outputs['instances'].get_fields())
  print('classes:',outputs['instances'].pred_classes.tolist())
  print('scores:', outputs['instances'].scores.tolist())
  print('pred_boxes:', (outputs['instances'].pred_boxes.tensor.tolist()))
  v = Visualizer(im[:, :, ::-1],
                metadata=None,
                scale=1
                 )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  #plt.imshow(out.get_image()[:, :, ::-1], cmap='gray', interpolation='bicubic')
  #plt.show()
  visualized_img = out.get_image()[:, :, ::-1];
  cv2.imwrite('visualized_img.jpg',visualized_img)
  cv2.imshow('visualized_img.jpg', visualized_img)
  cv2.waitKey(0)
cv2.destroyAllWindows()