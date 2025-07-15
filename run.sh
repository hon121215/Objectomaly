# SMIYC AnomalyTrack 
CUDA_VISIBLE_DEVICES=1 python3 anomaly_utils/anomaly_inference.py \
    --input '/root/Objectomaly/datasets/RoadAnomaly21/images/*.png' \
    --config-file '/root/Objectomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' \
    --output '/root/Objectomaly/results/RoadAnomaly21_anomaly'

# SMIYC RoadObsticle
CUDA_VISIBLE_DEVICES=1 python3 anomaly_utils/anomaly_inference.py \
    --input '/root/Objectomaly/datasets/RoadObsticle21/images/*.webp' \
    --config-file '/root/Objectomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' \
    --output '/root/Objectomaly/results/RoadObsticle21_anomaly'

# fs_static
# CUDA_VISIBLE_DEVICES=1 python3 anomaly_utils/anomaly_inference.py \
#     --input '/root/Objectomaly/datasets/fs_static/images/*.jpg' \
#     --config-file '/root/Objectomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' \
#     --output '/root/Objectomaly/results/fs_static_anomaly' 

# RoadAnomaly
# CUDA_VISIBLE_DEVICES=1 python3 anomaly_utils/anomaly_inference.py \
#     --input '/root/Objectomaly/datasets/RoadAnomaly/images/*.jpg' \
#     --config-file '/root/Objectomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' \
#     --output '/root/Objectomaly/results/RoadAnomaly_anomaly'
