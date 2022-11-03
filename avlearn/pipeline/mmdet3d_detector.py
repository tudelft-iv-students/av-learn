from typing import Optional, Union

from avlearn.datasets.detection.detdataset import DetectionDataset
from avlearn.modules.detectors.mmdet3d.apis.inference import init_model
from avlearn.modules.detectors.mmdet3d.apis.train import train_model
from mmcv import Config


class MMDet3DDetector:
    def __init__(self, cfg: Union[str, Config], 
                 checkpoint: Optional[str] = None, 
                 device: str = 'cuda:0') -> None:
        """
        :param cfg: Config dict or path to config file. 
                    Config dict should at least contain the key "type".
        :param checkpoint: Checkpoint path. If left as None, the model
                    will not load any weights. Defaults to None.
        :param device: Device used for inference. Defaults to 'cuda:0'.
        """
        # TODO: check if validating whether 
        # checkpoint belongs to cfg is required
        self.model, self.cfg = init_model(cfg, checkpoint, device, return_config=True)

    def forward(self):
        raise NotImplementedError
        
    def train(self, **kwargs):
        """A function wrapper for launching model training according to cfg."""
        dataset = DetectionDataset(self.cfg)
        distributed: bool = kwargs.get('distributed', False)
        validate: bool = kwargs.get('distributed', False)
        timestamp: Optional[str] = kwargs.get('timestamp', None)
        meta: Optional[dict] = kwargs.get('meta', None)       
        train_model(self.model, dataset, self.cfg, distributed, 
                    validate, timestamp, meta)
        
    def __call__(self):
        return self.forward()