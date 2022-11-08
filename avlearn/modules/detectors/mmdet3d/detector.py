from __future__ import division

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint
from mmdet import __version__ as mmdet_version
from mmdet.datasets import replace_ImageToTensor
from tqdm import tqdm

from avlearn.datasets.detection.mmdet3d_datasets import (build_dataloader,
                                                         build_dataset)
from avlearn.datasets.detection.mmdet3d_datasets.pipelines import Compose
from avlearn.modules.__base__ import BaseDetector
from avlearn.modules.detectors.mmdet3d import __version__ as mmdet3d_version
from avlearn.modules.detectors.mmdet3d.apis.inference import (
    show_det_result_meshlab, show_proj_det_result_meshlab)
from avlearn.modules.detectors.mmdet3d.apis.test import single_gpu_test
from avlearn.modules.detectors.mmdet3d.apis.train import (init_random_seed,
                                                          set_random_seed,
                                                          train_detector)
from avlearn.modules.detectors.mmdet3d.core.bbox import get_box_type
from avlearn.modules.detectors.mmdet3d.models import build_model
from avlearn.modules.detectors.mmdet3d.utils import (collect_env,
                                                     get_root_logger,
                                                     setup_multi_processes)
from avlearn.modules.detectors.mmdet3d.utils.detector import (
    convert_SyncBN, update_data_paths)


class MMDet3DDetector(BaseDetector):
    def __init__(
            self, cfg: Config, checkpoint: str = None,
            model_name=None, device=None) -> None:
        """Initialize an MMDetection3D detector.

        :param cfg: Detector configurations.
        :param checkpoint: Path to a checkpoint.
        :param model_name: Name of the initialized model, for logging purposes.
        :param device: Device to use (currently only GPU is supported).
        """
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.model = None

        if device is not None:
            self.device = device

        if not torch.cuda.is_available():
            raise NotImplementedError(
                "Currently, we only support cuda enabled environments")
        else:
            self.device = "cuda:0"

        self.device = torch.device(self.device)
        torch.cuda.set_device(self.device)

    def _build_model(self) -> None:
        self.model = build_model(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg'))
        self.model.init_weights()

        if self.checkpoint is not None:
            checkpoint = load_checkpoint(
                self.model, self.checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint['meta']:
                self.model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                self.model.CLASSES = self.cfg.class_names

        self.model.cfg = self.cfg
        self.model.to(self.device)

    def __init_logger(self, timestamp: str) -> None:
        """Initialize a logger.

        :param timestamp: Time of initialization.
        """
        log_file = self.cfg.work_dir / Path(f'{timestamp}.log')

        logger_name = 'mmdet3d' if self.model_name is None else self.model_name
        self.logger = get_root_logger(
            log_file=str(log_file),
            log_level=self.cfg.log_level, name=logger_name)

    def __modify_cfg(
            self,
            dataroot: Union[str, Path],
            work_dir: Union[str, Path] = None,
            epochs: int = None,
            batch_size: int = None,
            gpu_ids: Union[int, List[int]] = 0,
            autoscale_lr: bool = False) -> None:
        """Modify the configuration file according to inputs.

        :param dataroot: Path to data.
        :param work_dir: Directory to save output.
        :param epochs: Number of epochs to train the network.
        :param batch_size: Number of samples per batch.
        :param gpu_ids: Ids of the GPUs to use.
        :param autoscale_lr: Whether to use autoscaling of learning rate. 
        """
        # Data path
        self.cfg.data_root = str(dataroot)
        update_data_paths(self.cfg, self.cfg.data_root)

        # Output directory
        if work_dir is not None:
            self.cfg.work_dir = work_dir
        elif self.cfg.get('work_dir', None) is None:
            self.cfg.work_dir = f"results/{self.model_name}/training/"

        # Create work dir if it does not already exist
        Path(self.cfg.work_dir).mkdir(parents=True, exist_ok=True)

        # Number of training epochs
        if epochs is not None:
            self.cfg.runner.max_epochs = epochs

        # Batch size
        if batch_size is not None:
            self.cfg.data.samples_per_gpu = batch_size

        # Resume training if checkpoint is set
        if self.checkpoint is not None:
            self.cfg.resume_from = self.checkpoint

        if isinstance(gpu_ids, list):
            self.cfg.gpu_ids = gpu_ids
        else:
            self.cfg.gpu_ids = [gpu_ids]

        if autoscale_lr:
            # Apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
            self.cfg.optimizer['lr'] = self.cfg.optimizer['lr'] * len(
                self.cfg.gpu_ids) / 8

        # Save config in work dir
        self.cfg.dump(str(Path(self.cfg.work_dir, self.cfg_file.name)))

    def __create_meta(self, env_info: str) -> Dict[str, Any]:
        """Create a meta dictionary."""
        self.meta = dict()
        self.meta['env_info'] = env_info
        self.meta['seed'] = self.cfg.seed
        self.meta['exp_name'] = str(self.cfg_file)

    def train(
            self,
            dataroot: Union[str, Path],
            work_dir: Union[str, Path] = None,
            epochs: int = 20,
            batch_size: int = 16,
            gpu_ids: Union[int, List[int]] = 0,
            autoscale_lr: bool = False,
            random_seed: int = None,
            deterministic: bool = False,
            validate: bool = True,
            **kwargs) -> None:
        """Train the detector.

        :param dataroot: Path to data.
        :param work_dir: Directory to save output.
        :param epochs: Number of epochs to train the network.
        :param batch_size: Number of samples per batch.
        :param gpu_ids: Ids of the GPUs to use.
        :param autoscale_lr: Whether to use autoscaling of learning rate.
        :param random_seed: A random seed to use for reproducibility.
        :param deterministic: Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
        :param validate: Whether to perform validation during training.
        """
        # Set multi-process settings
        setup_multi_processes(self.cfg)

        # Set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # Add args to cfg
        self.__modify_cfg(dataroot, work_dir, epochs,
                          batch_size, gpu_ids, autoscale_lr)

        if self.model is None:
            self._build_model()

        self.model.train()

        # Initialize a logger
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.__init_logger(timestamp)

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        self.logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                         dash_line)

        # log some basic info
        self.logger.info(f'Config:\n{self.cfg.pretty_text}')

        # set random seeds
        if random_seed is not None:
            self.cfg.seed = random_seed
        else:
            self.cfg.seed = init_random_seed()
        set_random_seed(self.cfg.seed, deterministic=deterministic)

        self.logger.info(f'Set random seed to {self.cfg.seed}, '
                         f'deterministic: {deterministic}')
        self.logger.info(f'Model:\n{self.model}')

        datasets = [build_dataset(self.cfg.data.train)]
        if len(self.cfg.workflow) == 2:
            val_dataset = copy.deepcopy(self.cfg.data.val)
            # in case we use a dataset wrapper
            if 'dataset' in self.cfg.data.train:
                val_dataset.pipeline = self.cfg.data.train.dataset.pipeline
            else:
                val_dataset.pipeline = self.cfg.data.train.pipeline
            val_dataset.test_mode = False
            datasets.append(build_dataset(val_dataset))
        if self.cfg.checkpoint_config is not None:
            # Save mmdet version, config file content and class names in
            # checkpoints as meta data
            self.cfg.checkpoint_config.meta = dict(
                mmdet_version=mmdet_version,
                mmdet3d_version=mmdet3d_version,
                config=self.cfg.pretty_text,
                CLASSES=datasets[0].CLASSES)

        # Create the meta dict to record some important information such as
        # environment info and seed, which will be logged
        self.__create_meta(env_info)

        self.model.CLASSES = datasets[0].CLASSES
        train_detector(
            self.model,
            datasets,
            self.cfg,
            validate=validate,
            timestamp=timestamp,
            meta=self.meta)

    def forward(
            self, point_cloud: Union[str, Path, dict],
            batch_size: int = 16) -> List:
        """Forward pass.

        :param point_cloud: Path to a pcd.bin file, a directory containing
            pcd.bin files, or a point cloud dictionary.
        :param batch_size: Number of samples per batch.
        """
        self.cfg.model.train_cfg = None
        self.cfg.data.samples_per_gpu = batch_size
        convert_SyncBN(self.cfg.model)

        if self.model is None:
            self._build_model()

        self.model.eval()

        if isinstance(point_cloud, str) or isinstance(point_cloud, Path):
            point_cloud = Path(point_cloud)

            if point_cloud.is_dir():
                filenames = list(point_cloud.glob("*pcd.bin"))
            else:
                filenames = [point_cloud]

            cfg = self.model.cfg

            # build the data pipeline
            test_pipeline = copy.deepcopy(cfg.data.test.pipeline)
            test_pipeline = Compose(test_pipeline)
            box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

            results = []
            for i in tqdm(range(len(filenames)//batch_size + 1)):
                batch = []
                for pcd in filenames[i*batch_size: (i+1)*batch_size]:
                    pcd = str(pcd)
                    # load from point clouds file
                    data = dict(
                        pts_filename=pcd,
                        box_type_3d=box_type_3d,
                        box_mode_3d=box_mode_3d,
                        # for ScanNet demo we need axis_align_matrix
                        ann_info=dict(axis_align_matrix=np.eye(4)),
                        sweeps=[],
                        # set timestamp = 0
                        timestamp=[0],
                        img_fields=[],
                        bbox3d_fields=[],
                        pts_mask_fields=[],
                        pts_seg_fields=[],
                        bbox_fields=[],
                        mask_fields=[],
                        seg_fields=[])

                    data = test_pipeline(data)
                    batch.append(data)

                data = collate(batch, samples_per_gpu=batch_size)

                if next(self.model.parameters()).is_cuda:
                    # scatter to specified GPU
                    data = scatter(data, [self.device.index])[0]
                else:
                    data['img_metas'] = data['img_metas'][0].data
                    data['points'] = data['points'][0].data

                with torch.no_grad():
                    result = self.model(return_loss=False,
                                        rescale=True, **data)

                results.append(result)

            return results

        elif isinstance(point_cloud, dict):
            # TODO: Add inference for loaded point clouds dicts
            raise NotImplementedError(
                "We currently only support inference from point cloud files "
                "or directories containing point cloud files."
            )

    def evaluate(self,
                 dataroot: Union[str, Path],
                 work_dir: Union[str, Path] = None,
                 batch_size: int = 16,
                 gpu_ids: Union[int, List[int]] = 0,
                 random_seed: int = None,
                 deterministic: bool = False) -> None:

        # Set multi-process settings
        setup_multi_processes(self.cfg)

        # Set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        self.cfg.model.train_cfg = None
        self.__modify_cfg(
            dataroot=dataroot,
            work_dir=work_dir,
            batch_size=batch_size,
            gpu_ids=gpu_ids)
        convert_SyncBN(self.cfg.model)

        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2,
            shuffle=False)

        # in case the test dataset is concatenated
        if isinstance(self.cfg.data.test, dict):
            self.cfg.data.test.test_mode = True
            if self.cfg.data.get(
                    'test_dataloader', {}).get(
                    'samples_per_gpu', 1) > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                self.cfg.data.test.pipeline = replace_ImageToTensor(
                    self.cfg.data.test.pipeline)
        elif isinstance(self.cfg.data.test, list):
            for ds_cfg in self.cfg.data.test:
                ds_cfg.test_mode = True
            if self.cfg.data.get(
                    'test_dataloader', {}).get(
                    'samples_per_gpu', 1) > 1:
                for ds_cfg in self.cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        test_loader_cfg = {
            **test_dataloader_default_args,
            **self.cfg.data.get('test_dataloader', {})
        }
        # build the dataloader
        dataset = build_dataset(self.cfg.data.test)
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        # set random seeds
        if random_seed is not None:
            set_random_seed(random_seed, deterministic=deterministic)

        if self.model is None:
            self._build_model()

        self.model.eval()

        self.model = MMDataParallel(self.model, device_ids=self.cfg.gpu_ids)
        results = single_gpu_test(self.model, data_loader)

        return dataset.format_results(
            results, jsonfile_prefix=self.cfg.work_dir)

    def visualize(self, data: dict, result: List[dict],
                  out_dir: str, score_thr: float = 0.0,
                  show: bool = True, snapshot: bool = False) -> None:
        """A function wrapper for visualizing the output of the detector model.

        :param data: Input points and the information of the sample.
        :param result: Prediction results.
        :param out_dir: Output directory of visualization result.
        :param score_thr: Minimum score of bboxes to be shown. Defaults to 0.0.
        :param show: Visualize the results online. Defaults to True.
        :param snapshot: Whether to save the online results. Defaults to False.
        """
        if self.model is None:
            self._build_model()

        if 'img' in data.keys():
            show_proj_det_result_meshlab(data, result, out_dir,
                                         score_thr, show, snapshot)
        else:
            show_det_result_meshlab(data, result, out_dir,
                                    score_thr, show, snapshot)

    def __call__(self, point_cloud: Union[str, Path, dict],
                 batch_size: int = 16) -> List:
        return self.forward(point_cloud, batch_size)
