# SMIYC AnomalyTrack 
CUDA_VISIBLE_DEVICES=1 python3 anomaly_utils/anomaly_inference.py \
    --input '/root/Mask2Anomaly/datasets/RoadAnomaly21/images/*.png' \
    --config-file '/root/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' \
    --output '/root/Mask2Anomaly/results/RoadAnomaly21_anomaly_250701_anomaly_score_map'

# SMIYC RoadObsticle
CUDA_VISIBLE_DEVICES=1 python3 anomaly_utils/anomaly_inference.py \
    --input '/root/Mask2Anomaly/datasets/RoadObsticle21/images/*.webp' \
    --config-file '/root/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' \
    --output '/root/Mask2Anomaly/results/RoadObsticle21_anomaly_250701_anomaly_score_map'

# fs_static
# CUDA_VISIBLE_DEVICES=1 python3 anomaly_utils/anomaly_inference.py \
#     --input '/root/Mask2Anomaly/datasets/fs_static/images/*.jpg' \
#     --config-file '/root/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' \
#     --output '/root/Mask2Anomaly/results/fs_static_anomaly_250620_inference' 

# RoadAnomaly
# CUDA_VISIBLE_DEVICES=1 python3 anomaly_utils/anomaly_inference.py \
#     --input '/root/Mask2Anomaly/datasets/RoadAnomaly/images/*.jpg' \
#     --config-file '/root/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' \
#     --output '/root/Mask2Anomaly/results/RoadAnomaly_anomaly_250620_inference'