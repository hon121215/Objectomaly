# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
from PIL import Image, ImageDraw, ImageChops
import torch.nn.functional as F
from easydict import EasyDict
import scipy.stats as st

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tempfile
import random
import time
import warnings
import torch
import cv2
import numpy as np
import tqdm
import torch.nn as nn
import math

import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import cm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision import transforms
from mask2former import add_maskformer2_config, SemanticSegmentorWithTTA
from demo.predictor import VisualizationDemo
from torch.autograd import Variable
from matplotlib import pyplot
from torchvision.utils import save_image
# import kornia as K

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# constants
WINDOW_NAME = "mask2former demo"

# Using SAM 
sam = sam_model_registry["vit_h"](checkpoint='/root/Objectomaly/sam_vit_h_4b8939.pth')
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = "cpu"  # "cuda" or "cpu"
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/root/Objectomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="/root/Objectomaly/Validation_Dataset/Validation_Dataset/RoadAnomaly21/images/*.png",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/root/Objectomaly/results/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--aug",
        default=True,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')
    
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    anomaly_score_list = []
    ood_gts_list = []

    inference_times = []
    if args.input:
        image_paths = glob.glob(os.path.expanduser(str(args.input[0])))
        
        for path in glob.glob(os.path.expanduser(str(args.input[0]))):
            start_time = time.time()
            img = read_image(path, format="BGR")
                        
            # img_ud = np.flipud(img) # Aug 1
            img_lr = np.fliplr(img) # Aug 2

            predictions_na, mask_info = demo.run_on_image(img)
            predictions_lr, mask_info2 = demo.run_on_image(img_lr)
                  
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions_na["instances"]))
                    if "instances" in predictions_na
                    else "finished",
                    time.time() - start_time,
                )
            )
            
            if args.output:

                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                    
                save_dir = os.path.join(args.output, "visuals")
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(path))[0]
    
                predictions_naa =  predictions_na["sem_seg"].unsqueeze(0)
                outputs_na = 1 - torch.max(predictions_naa[0:19,:,:], axis = 1)[0]
                outputs_na = outputs_na.detach().cpu().numpy().squeeze()

                predictions_lrr =  predictions_lr["sem_seg"].unsqueeze(0)
                outputs_lr = 1 - torch.max(predictions_lrr[0:19,:,:], axis = 1)[0]
                outputs_lr = outputs_lr.detach().cpu().numpy().squeeze()
                outputs_lr = np.flip(outputs_lr, axis=1)
                
                outputs = np.expand_dims((outputs_lr + outputs_na )/2.0, 0).astype(np.float32)
                
                sam_result = mask_generator.generate(img)
                masks = [mask['segmentation'] for mask in sorted(sam_result, key=lambda x: x['area'], reverse=False)]
                updated_outputs_sam = outputs.copy()
                for mask in masks:
                    resized_mask = cv2.resize(mask.astype(np.uint8), (outputs.shape[2], outputs.shape[1]), interpolation=cv2.INTER_NEAREST)
                    resized_mask = resized_mask.astype(bool)
                    mask_mean_value = np.mean(outputs[0][resized_mask])
                    updated_outputs_sam[0][resized_mask] = mask_mean_value

                updated_outputs_laplacian = updated_outputs_sam.copy()
                edges = cv2.Laplacian(updated_outputs_laplacian[0], cv2.CV_32F)
                for i in range(1, updated_outputs_laplacian.shape[1] - 1):
                    for j in range(1, updated_outputs_laplacian.shape[2] - 1):
                        if edges[i, j] > 0:
                            surrounding_values = updated_outputs_laplacian[0][i-1:i+2, j-1:j+2][edges[i-1:i+2, j-1:j+2] == 0]
                            if surrounding_values.size > 0:
                                updated_outputs_laplacian[0][i, j] = np.mean(surrounding_values)

                updated_outputs_final = updated_outputs_laplacian.copy()
                updated_outputs_final[0] = cv2.GaussianBlur(updated_outputs_final[0], (7, 7), sigmaX=1, sigmaY=1)
                
                anomaly_map = updated_outputs_final[0]
                anomaly_map_norm = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
                anomaly_map_uint8 = (anomaly_map_norm * 255).astype(np.uint8)

                anomaly_map_color = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
                
                save_path = os.path.join(save_dir, f"{base_name}_anomaly.jpg")
                cv2.imwrite(save_path, anomaly_map_color)
                print(f"[INFO] Anomaly map saved at: {save_path}")

                pathGT = path.replace("images", "labels_masks")                
               
                if "RoadObsticle21" in pathGT:
                   pathGT = pathGT.replace("webp", "png")
                if "fs_static" in pathGT:
                   pathGT = pathGT.replace("jpg", "png")                
                if "RoadAnomaly" in pathGT:
                   pathGT = pathGT.replace("jpg", "png") 
                
                mask = Image.open(pathGT)
                ood_gts = np.array(mask)

                if "RoadAnomaly" in pathGT:
                    ood_gts = np.where((ood_gts==2), 1, ood_gts)
                if "LostAndFound" in pathGT:
                    ood_gts = np.where((ood_gts==0), 255, ood_gts)
                    ood_gts = np.where((ood_gts==1), 0, ood_gts)
                    ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

                if "Streethazard" in pathGT:
                    ood_gts = np.where((ood_gts==14), 255, ood_gts)
                    ood_gts = np.where((ood_gts<20), 0, ood_gts)
                    ood_gts = np.where((ood_gts==255), 1, ood_gts)

                if 1 not in np.unique(ood_gts):
                    continue              
                else:
                    ood_gts_list.append(np.expand_dims(ood_gts, 0))          
                    anomaly_score_list.append(updated_outputs_final)


    file.write( "\n")
    ood_gts = np.stack(ood_gts_list, axis=0)
    anomaly_scores = np.stack(anomaly_score_list, axis=0)
    
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)
    
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    fpr, tpr, _ = roc_curve(val_label, val_out)    
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    
    def compute_confusion_matrix(pred_mask, gt_mask):
        # True Positive, False Positive, False Negative, True Negative
        TP = np.sum((pred_mask == 1) & (gt_mask == 1))
        FP = np.sum((pred_mask == 1) & (gt_mask == 0))
        FN = np.sum((pred_mask == 0) & (gt_mask == 1))
        TN = np.sum((pred_mask == 0) & (gt_mask == 0))
        return TP, FP, FN, TN

    def calc_metrics_from_confmat(tp, fp, fn, tn):
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return iou, precision, f1
    
    fixed_threshold = np.arange(0.7, 0.951, 0.01)
    
    for thresh in fixed_threshold:
        pred_mask_fixed = (anomaly_scores >= thresh).astype(np.uint8)
        gt_mask = (ood_gts == 1).astype(np.uint8)
        TP, FP, FN, TN = compute_confusion_matrix(pred_mask_fixed, gt_mask)
        iou_fixed, ppv_fixed, f1_fixed = calc_metrics_from_confmat(TP, FP, FN, TN)

    print(f'AUPRC score: {prc_auc}')
    print(f'FPR@TPR95: {fpr}')
    print(f'sIoU: {iou_fixed}')
    print(f'PPV (Precision): {ppv_fixed}')
    print(f'F1-Score: {f1_fixed}')
    
    file.write(f'\nDataset: {args.output} Threshold: {thresh:.2f} AUPRC: {prc_auc} FPR@TPR95: {fpr} sIoU: {iou_fixed} PPV (Precision): {ppv_fixed} F1-Score: {f1_fixed}')
    file.close()