# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from detectron2.data import MetadataCatalog, Metadata
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        vis_output = visualizer.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))

        # if "panoptic_seg" in predictions:
        #     panoptic_seg, segments_info = predictions["panoptic_seg"]
        #     vis_output = visualizer.draw_panoptic_seg_predictions(
        #         panoptic_seg.to(self.cpu_device), segments_info
        #     )
        # else:
        #     if "sem_seg" in predictions:
        #         vis_output = visualizer.draw_sem_seg(
        #             predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
        #         )
        #     if "instances" in predictions:
        #         instances = predictions["instances"].to(self.cpu_device)
        #         vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output