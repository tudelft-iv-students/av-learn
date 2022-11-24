# Experiments
In an attempt to address research questions related to AV pipelines, we conducted specific experiments using AV Learn.

## Research Questions
The AV Learn framework was utilised to shed light on certain aspects of the AV objective and answer a range of different research questions:
- How do learning-based trackers perform as compared to non-learning-based trackers?
- How does the performance on an upstream task correlate with the performance on a downstream task?
- How do the sequential and mixed multitask (Detection
- Tracking) paradigms compare in terms of end-to-end performance?

## Networks
### CenterPoint
CenterPoint [[1]](#1) was developed to solve the problem occurring by traditional 3D box representation approaches, which do not reflect objects 
in the 3D world and lack any particular orientation. Hence, CenterPoint represents, detects, and tracks 3D objects as points. The first step involves 
the detection of the centers of objects, followed by various regression heads to compute other attributes. These heads include 3D size, 3D orientation, 
and velocity, with the latter being utilized to compute the object tracks by leveraging a greedy closest-point matching. Our motivation for utilizing 
CenterPoint comes from the architecture's inherit support for both detection and tracking, and could thus be introduced in our mixed-multitask pipelines.

### Kalman Filter
The Kalman Filter [[2]](#2) is one of the most important and common estimation algorithms, used to produce estimates of hidden variables, 
based on inaccurate and uncertain measurements. Our Kalman tracker implementation is based on the AB3DMOT tracker [[3]](#3), which combines 
a 3D Kalman filter with the Hungarian algorithm to perform state estimation and detection - tracking association. Our motivation for choosing a Kalman-based 
tracker lies in the fact that it constitutes an extensively tested, robust algorithm, which could act as a tracking baseline for our framework. Additionally, 
since the Kalman tracker is a non-learning based tracker it could be compared with the learning-based CenterPoint tracker to facilitate our second research 
question.

### LaneGCN
LaneGCN's [[4]](#4) novelty lies in the construction of a lane graph (of the centerlines) from raw map representations, capable of preserving 
the map structure. The network's architecture extends graph convolutions with multiple adjacency matrices and along-lane dilations. Subsequently, a fusion 
network is utilized to capture the interactions between actor-to-lane, lane-to-lane, lane-to-actor and actor-to-actor. Our motivation for choosing LaneGCN 
as a prediction module, was its ability to predict accurate and realistic future trajectories. While LaneGCN has been tested and proved to be working well 
in the case of the [Argoverse](https://www.argoverse.org/) dataset, no official implementation exists for nuScenes. In order to evaluate its performance, 
we constructed a graph representation of the centerlines (arcline paths) of the maps in nuScenes. These paths were discretized to lane nodes and fed to a map 
representation used in the Graph Convolutional Network (GCN) of LaneGCN.

## Individual Task Evaluation
In this experiment we studied the performance of each task in isolation, *i.e.*, when being passed a ground-truth input and not the inference results of 
previous stages of the pipeline.

### Detection
Detection corresponds to the first task of the pipeline. Its evaluation is therefore directly in the ground-truth setting. For the purposes of this experiment, 
two CenterPoint [[1]](#1) networks were trained for *10* epochs on the full nuScenes training set, with the first only using the detection head, and 
the second using both the detection and velocity heads. The two different detectors were evaluated on the nuScenes  
[detection task](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any). Checkpoints for the two trained CenterPoint variants 
were used to perform inference on the 6019 sample tokens of the nuScenes validation set. The following table illustrates the detection results of the 
two different CenterPoint networks on the nuScenes validation set, along with 2 baselines.

|               |  mAP  | mATE (m) | mASE (1-IoU) | mAOE (rad) | mAVE (m/s) | mAAE (1-acc) |  NDS  |
|---------------|:-----:|:--------:|:------------:|:----------:|:----------:|:------------:|:-----:|
| CenterPoint*1 | 0.480 |   0.308  |     0.264    |    0.409   |    1.193   |     0.446    | 0.497 |
| CenterPoint   | 0.469 |   0.311  |     0.268    |    0.432   |    0.388   |     0.197    | 0.575 |
|               |       |          |              |            |            |              |       |
| Megvii        | 0.519 |   0.300  |     0.247    |    0.379   |    0.245   |     0.14     | 0.628 |
| PointPillars  | 0.295 |   0.517  |     0.290    |    0.500   |    0.416   |     0.368    | 0.448 |

1: CenterPoint* indicates the CenterPoint variant trained only with the detection head

It becomes evident that the Megvii [[5]](#5) baseline leads to the best detection performance, with our 2 models outperforming 
the PointPillars baseline, yet these results might not be representative. Regarding the comparison of the two variants, it is important 
to note that the network without the velocity head (CenterPoint*), does not output velocities. This leads to a seemingly high velocity error (**mAVE**), 
which should not be taken into account. This problem is also reflected by the lower value of the \textit{NDS} metric, which is a weighted average of the rest of 
the metrics (including the velocity error). However, the network is capable of overall producing more accurate bounding-box positions, as illustrated by the 
metrics **mAP**, **mATE** and **mAOE**. This finding is rather intuitive since the network is trained exclusively on the detection task, with the 
velocity error not affecting the overall objective, and thus the training progress.

### Tracking
For tracking, we extracted the ground-truth detections from the nuScenes validation set, which were in turn passed through either the CenterPoint or the Kalman 
tracker. The table below shows the tracking results for both modules, with respect to the AMOTA, AMOTP, and Recall.

|                     | AMOTA | AMOTP | Recall |
|---------------------|:-----:|:-----:|:------:|
| Kalman Tracker      | 0.789 | 0.405 |  0.859 |
| CenterPoint Tracker | 0.871 | 0.211 |  0.964 |

We notice that the CenterPoint tracker is able to outperform the Kalman tracker across all metrics. We argue that the CenterPoint tracker is able to attain 
these better results, by exploiting the ground-truth velocity information, which the Kalman tracker fails to take into account. Kalman filter instead only 
relies on the detections to infer the velocities, which are in turn used to associate the objects to tracks.

### Prediction
The final part of this experiment involves the evaluation of the LaneGCN trained model on the nuScenes validation set for the prediction task. 
In this setting the ground-truth past trajectories of instance - sample token pairs are given as an input to the trained model. 

We compare LaneGCN with 2 official submissions on the nuScenes prediction challenge for reference: Trajectron++ [[6]](#6) and 
LaPred [[7]](#7). The evaluation results, shown in the following tables, indicate that the prediction model 
does indeed learn how to predict the future trajectories of agents, being able to outperform Trajectron++ in most metrics, but performing worse than the more 
competitive LaPred network. Before discussing the results, we should mention that in the AV prediction task there is no universally accepted evaluation metric. 
This leads to the usage of a range of different metrics with their strengths and weaknesses. That being said, LaneGCN performs comparable with top-performing 
models in the prediction challenge for the metrics **MinADE_10, MissRateTopK_2_5, MissRateTopK_2_10, and OffroadRate**, while its results are 
adequate for the metrics **Min_ADE_5** and **MinFDE_1**. It is yet to be determined whether simple modifications in the network architecture and a more 
extensive fine-tuning of the model's hyperparameters, along with the introduction of more features of nuScenes, could improve its overall performance.


|               | MinADE_5 | MinADE_10 | MissRateTopK_2_5 | MissRateTopK_2_10 | MinFDE_1 | OffroadRate |
|---------------|:--------:|:---------:|:----------------:|:-----------------:|:--------:|:-----------:|
| LaneGCN       |   2.289  |   1,318   |      63.54%      |       50.74%      |   9.148  |    0.052    |
|               |          |           |                  |                   |          |             |
| Trajectron++2 |   1,877  |   1,510   |      69.76%      |       56.8%       |   9.517  |    0.250    |
| LaPred2       |   1,235  |   1,054   |       52.6%      |       46.07%      |   8.368  |    0.091    |


## End-to-end performance
In this experiment we calculated the performance of the whole AV pipeline on the prediction task (**i.e.,** we produced detections from raw input data, 
then trackings, and finally predictions)

### Setup
To perform inference up to the prediction task, the 6019 sample tokens of the nuScenes validation set were passed 
through the CenterPoint model, producing the detections. These were utilized by the CenterPointTracker to produce the object tracks, 
which in turn were used as input to the prediction module (**i.e.**, LaneGCN).

### Instance Token Association Problem
The official nuScenes prediction challenge evaluates the performance of predicted trajectories on selected sample token (**i.e.**, a 
specific sample of a scene) - instance token (**i.e.**, a specific instance of an object in a sample of a scene) pairs. These pairs are 
directly accessible if we want to directly perform inference on the ground-truth past trajectories of these agents, similarly to our first experiment. 
However, in the case of inferring the predictions from the entire pipeline (**i.e.**, from the inferred tracked trajectories of our tracking pipelines), 
there is no direct way to associate an instance - sample token pair to a sample token - track id pair. In order to make these associations, for each instance 
- sample token, whose future trajectories we needed to infer, we assumed that the track id, whose translation coordinates are the closest to the instance token's 
current position, corresponded to the same agent as the instance token. It should be noted that this process does not guarantee correct associations, since it is 
prone to false positives. We expected for it to have a degrading effect on the prediction performance of the inference pipeline.

### Prediction Results
When evaluating the results on the nuScenes dataset for both the pipeline, 
we noticed that the values of all evaluation metrics are bad, with **Min_ADE_5 ~ 19** and **MissRateTopK_2_5 ~ 98%**. This finding illustrates 
that possible erroneous associations between tracking ids and instance tokens are combined with the accumulated error rates from upstream tasks (**i.e.**, the 
error rates of detection and tracking), leading to a collapse of the pipeline's performance on the prediction task. Both the sequential and mixed-multitask pipelines 
appear to be prone to this aggregation of error rates.

| MinADE@5 | MinADE@10 | MinFDE@1 | OffRoadRate | MissRate@5 | MissRate@10 |
|:--------:|:---------:|:--------:|:-----------:|:----------:|:-----------:|
|  18.654  |   18.304  |  44.603  |    0.477    |   98.23%   |    98.06%   |



## References
<a id="1">[1]</a> Yin, Tianwei, Xingyi Zhou, and Philipp Krahenbuhl. "Center-based 3d object detection and tracking." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

<a id="2">[2]</a> Kalman, Rudolph Emil. "A new approach to linear filtering and prediction problems." (1960): 35-45.

<a id="3">[3]</a> Weng, Xinshuo, et al. "Ab3dmot: A baseline for 3d multi-object tracking and new evaluation metrics." arXiv preprint arXiv:2008.08063 (2020).

<a id="4">[4]</a> Liang, Ming, et al. "Learning lane graph representations for motion forecasting." European Conference on Computer Vision. Springer, Cham, 2020.

<a id="5">[5]</a> Simonelli, Andrea, et al. "Disentangling monocular 3d object detection." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

<a id="6">[6]</a> Salzmann, Tim, et al. "Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data." European Conference on Computer Vision. Springer, Cham, 2020.

<a id="7">[7]</a> Kim, ByeoungDo, et al. "Lapred: Lane-aware prediction of multi-modal future trajectories of dynamic agents." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
