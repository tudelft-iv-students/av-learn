import inspect
from typing import Optional, Union

from avlearn.datasets.detection.detdataset import DetectionDataset
from avlearn.modules.detectors.mmdet3d.apis.inference import \
    init_model, show_det_result_meshlab, show_proj_det_result_meshlab
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
        
    def visualize(self, data: dict, result: dict, 
                  out_dir: str, score_thr: float = 0.0, 
                  show: bool = False, snapshot: bool = False):
        """
        :param data (dict): Input points and the information of the sample.
        :param result (dict): Prediction results.
        :param out_dir (str): Output directory of visualization result.
        :param score_thr (float, optional): Minimum score of bboxes to be shown.
                                            Defaults to 0.0.
        :param show (bool, optional): Visualize the results online. 
                                      Defaults to False.
        :param snapshot (bool, optional): Whether to save the online results.
                                          Defaults to False.
        """
        method = getattr(self.model, 'show_results', None)
        if callable(method):
            # Since show_results may be uniquely implemented for 
            # each model, pass only those parameters that are actually 
            # valid for this specific implementation.
            valid_keys = inspect.signature(self.model.show_results).parameters.keys()
            kwargs = {key:value for key, value in locals().items() if key in valid_keys}
            self.model.show_results(**kwargs)
        elif 'img' in data.keys():
            show_proj_det_result_meshlab(data, result, out_dir, 
                                         score_thr, show, snapshot)
        else:
            show_det_result_meshlab(data, result, out_dir, 
                                    score_thr, show, snapshot)
        
    def __call__(self):
        return self.forward()