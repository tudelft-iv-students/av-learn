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
