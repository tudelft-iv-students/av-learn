# Prerequisites
In this section we demonstrate how to prepare an environment with PyTorch. AV-Learn has been tested and works on Linux, and requires the following packages:
- Python 3.7+
- PyTorch 1.3+
- CUDA 9.2+
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name avlearn python=3.7 -y
conda activate avlearn
```

**Step 2.** Install necessary packages
```
git clone https://github.com/nightrome/av-learn
cd av-learn/

pip install -r requirements.txt

mim install mmcv-full
mim install mmdet

pip install cumm-cuXXX && pip install spconv-cuXXX
```

Where XXX is the CUDA version that you use. \
E.g. `pip install cumm-cu113 && pip install spconv-cu113` for CUDA 11.3.

# Dataset Preparation
## Before Preparation
It is recommended to use `$av-learn/data` as the root folder for all the datasets.

## Download and Data Preparation

### NuScenes

Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running

```bash
python avlearn/datasets/tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

In order to use the LaneGCN network for the prediction task, you also need map representations for nuScenes. 

First, you have to download the map expansion package from [nuScenes](https://www.nuscenes.org/download). Then, you could either create the map representations using the following script:
```bash
python avlearn/datasets/tools/create_nuscenes_graph.py --data_dir ./data/nuscenes --output_dir ./data/nuscenes_representations --maps_dir ./data/nuscenes/maps
```

or download them from [here](https://drive.google.com/drive/folders/1--28wIYgFBrpG_IxkG04OVhH7dxf6v_B?usp=share_link).



# Initialize a pipeline
You can initialize an AV Learn pipeline using a list of modules. The list does not need to contain modules for all of the corresponding tasks of the pipeline, unless end-to-end evaluation is to be performed. In the examples below we initialize different pipelines.

## Example 1
In this pipeline we use a CenterPointDetector, a non-learning based CenterPointTracker, and the LaneGCN predictor.

```python
from avlearn.pipeline import Pipeline
from avlearn.modules.detectors import CenterPointDetector
from avlearn.modules.trackers import CenterPointTracker
from avlearn.modules.predictors import LaneGCN

detector = CenterPointDetector()
tracker = CenterPointTracker()
predictor = LaneGCN()

pipeline = Pipeline([detector, tracker, predictor])
```

## Example 2
In this pipeline we use the KalmanTracker for tracking. Thus the velocity head of the CenterPointDetector can be disabled (optional). 
```python
from avlearn.pipeline import Pipeline
from avlearn.modules.detectors import CenterPointDetector
from avlearn.modules.trackers import KalmanTracker
from avlearn.modules.predictors import LaneGCN

detector = CenterPointDetector(velocity=False)
tracker = KalmanTracker()
predictor = LaneGCN()

pipeline = Pipeline([detector, tracker, predictor])
```

## Example 3
In this pipeline we only define a predictor. We can still use the pipeline for training and individual evaluation of the predictor, but the end-to-end evaluation is disabled.
```python
from avlearn.pipeline import Pipeline
from avlearn.modules.predictors import LaneGCN

predictor = LaneGCN()

pipeline = Pipeline([predictor])
```

# Training
The AV Learn framework simplifies the process of trainning the corresponding modules of a pipeline. Using the example below you can train all of the pipeline's trainable modules.

```python
pipeline.train(
    dataroot="./data/nuscenes",
    work_dir="./results/training",
    n_epochs=[20, 0, 36],
    batch_size=[8, 0, 4],
    map_dataroot="./data/nuscenes_representations" # Model specific (LaneGCN)
)
```

# Evaluation
Once you obtain the pretrained models, and their checkpoints, evaluating them either individually, or the end-to-end performance of the pipeline, can be performed using the following block of code.

```python
detector = CenterPointDetector(checkpoint="/path/to/checkpoint/")
tracker = CenterPointTracker()
predictor = LaneGCN(checkpoint_pth="/path/to/checkpoint/")

pipeline = Pipeline([detector, tracker, predictor])

pipeline.evaluate(
    dataroot="./data/nuscenes",
    work_dir="./results/evaluation",  
    end_to_end=False, # or set True for end-to-end evaluation
    map_dataroot="./data/nuscenes_representations" # Model specific (LaneGCN)
)
```