# Nuscenes classes for which the tracklets are calculated
NUSCENES_TRACKING_CLASSES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


# 99.9 percentile of the l2 velocity error distribution (per class / 0.5 second)
# This is an earlier basci statistc. Tuning this could provide some considerable
#  AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
    'car': 4,
    'truck': 4,
    'bus': 5.5,
    'trailer': 3,
    'pedestrian': 1,
    'motorcycle': 13,
    'bicycle': 3,
}
