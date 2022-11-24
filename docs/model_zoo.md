# Model Zoo

## Detection
For the detection task we trained the CenterPoint network on the nuScenes dataset, with and without the velocity head, for 10 epochs each. 

|               |  mAP  | mATE (m) | mASE (1-IoU) | mAOE (rad) | mAVE (m/s) | mAAE (1-acc) |  NDS  | checkpoint | config    |
|---------------|:-----:|:--------:|:------------:|:----------:|:----------:|:------------:|:-----:|------------|-----------|
| CenterPoint*1 | 0.480 |   0.308  |     0.264    |    0.409   |    1.193   |     0.446    | 0.497 | [model](https://drive.google.com/file/d/1zH3jpTh4-m8digFV56HRT1WOVB-ymCgp/view?usp=share_link)      | [config.py](https://drive.google.com/file/d/1Aq_44WaI4om-D6_0QFyDERVwJDC9apfC/view?usp=share_link) |
| CenterPoint   | 0.469 |   0.311  |     0.268    |    0.432   |    0.388   |     0.197    | 0.575 | [model](https://drive.google.com/file/d/1Bi_IJEh5Df5_v6mtCNFUQSLnShSr1act/view?usp=share_link)      | [config.py](https://drive.google.com/file/d/1MY_q_vvkxv1LgR7xB5M6m3eZky4_ZvPg/view?usp=share_link) |

1: CenterPoint* indicates the CenterPoint variant trained only with the detection head

Further training is required to achieve the performance obtained by [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint/README.md)

## Tracking
For the tracking task we used the non-learning-based algorithm Kalman Filter, and a CenterPointTracker which computes trackings based on the output velocities of the CenterPoint detection network (with the velocity head). Thefore, we did not train a network for this task.

## Prediction
For the prediction task we trained the LaneGCN network on the nuScenes dataset for 36 epochs.

|         | MinADE_5 | MinADE_10 | MissRateTopK_2_5 | MissRateTopK_2_10 | MinFDE_1 | OffroadRate | checkpoint |
|---------|:--------:|:---------:|:----------------:|:-----------------:|:--------:|:-----------:|------------|
| LaneGCN |   2.289  |   1,318   |      63.54%      |       50.74%      |   9.148  |    0.052    | [model](https://drive.google.com/file/d/1oNSUbofQOrjKJaXkc0XF7GlbioNf42IR/view?usp=share_link)      |
