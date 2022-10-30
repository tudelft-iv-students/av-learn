# Prerequisites
In this section we demonstrate how to prepare an environment with PyTorch. AV-Learn works on Linux, Windows and macOS and requires the following packages:
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

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```



# Installation
```
git clone https://github.com/nightrome/av-learn
cd av-learn/requirements
pip install -r requirements.txt
```


# Dataset Preparation
## Before Preparation

It is recommended to use `$av-learn/data` as the root folder for all the datasets.
If your folder structure is different from the following, you may need to change the corresponding paths in config files.

```
mmdetection3d
├── avlearn
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval

```

## Download and Data Preparation

### NuScenes

Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running

```bash
python avlearn/datasets/tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```