import warnings
from pathlib import Path
from typing import List, Union

from utils import (check_missing_args, print_pipeline_info,
                   warn_module_replacement)

from avlearn.modules.__base__ import BaseDetector, BasePredictor, BaseTracker


class Pipeline:
    def __init__(self, modules: List) -> None:
        self.detector = None
        self.tracker = None
        self.predictor = None

        if not isinstance(modules, list):
            modules = [modules]

        for module in modules:
            if issubclass(module.__class__, BaseDetector):
                warn_module_replacement(self.detector, module, "detector")
                self.detector = module

            elif issubclass(module.__class__, BaseTracker):
                warn_module_replacement(self.tracker, module, "tracker")
                self.tracker = module

            elif issubclass(module.__class__, BasePredictor):
                warn_module_replacement(self.predictor, module, "predictor")
                self.predictor = module

            else:
                warnings.warn(
                    "All modules used should inherit from AV Learn's base "
                    f"classes (avlearn.modules.__base__). {module.__class__} "
                    "will not be used.")

        self.modules = [self.detector, self.tracker, self.predictor]
        print_pipeline_info(self.modules)

    def train(
        self,
        dataroot: Union[str, Path],
        work_dir: Union[str, Path] = None,
        n_epochs: Union[int, List[int]] = None,
        batch_size: Union[int, List[int]] = 8,
        gpu_ids: Union[int, List[int]] = 0,
        random_seed: int = None,
        **kwargs
    ):
        args = locals()
        del args['self']

        for module in self.modules:
            if module is None:
                continue

            train_method = getattr(module, "train", None)
            if callable(train_method):
                check_missing_args(args, train_method)
                module.train(**args)
            else:
                warnings.warn(
                    f"Module {module.__class__.__name__} is not trainable, or "
                    "it does not have a train method."
                )

    def evaluate(self):
        pass

    def forward(self):
        pass
