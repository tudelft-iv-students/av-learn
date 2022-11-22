import json
import warnings
from pathlib import Path
from typing import List, Union

from avlearn.apis.evaluate import Evaluator
from avlearn.modules.__base__ import BaseDetector, BasePredictor, BaseTracker

from .utils import (check_missing_args, cpu, format_for_tracking,
                    print_pipeline_info, warn_module_replacement)


class Pipeline:
    def __init__(self, modules: List) -> None:
        """Initialize an AV Learn pipeline.

        :param modules: List of modules to use in the pipeline.
        """
        self.detector = None
        self.tracker = None
        self.predictor = None
        self.n_modules = len(modules)

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
        **kwargs
    ) -> None:
        """Train the pipeline's modules individually.

        :param dataroot: Root path to data dir.
        :param work_dir: Path to output dir.
        :param n_epochs: Number of training epochs.
        :param batch_size: Number of samples in each batch.
        :param gpu_ids: Which gpus to use.

        Note: Model-specific arguments can also be given. 
        (e.g. map_dataroot for LaneGCN)
        """
        args = locals()
        del args['self']
        del args['batch_size']
        del args['n_epochs']

        if isinstance(batch_size, int):
            batch_size = [batch_size] * 3
        elif isinstance(batch_size, list):
            assert len(batch_size) != self.n_modules, (
                f"Pipeline was initialized with {self.n_modules} "
                f"modules, but only {len(batch_size)} batch sizes were given.")

            if self.detector is None:
                batch_size.insert(0, None)
            if self.tracker is None:
                batch_size.insert(1, None)
            if self.predictor is None:
                batch_size.append(None)

        if isinstance(n_epochs, int):
            n_epochs = [n_epochs] * 3
        elif isinstance(n_epochs, list):
            assert len(n_epochs) != self.n_modules, (
                f"Pipeline was initialized with {self.n_modules} modules, "
                f"but only {len(n_epochs)} number of epochs were given.")

            if self.detector is None:
                n_epochs.insert(0, None)
            if self.tracker is None:
                n_epochs.insert(1, None)
            if self.predictor is None:
                n_epochs.append(None)

        for i, module in enumerate(self.modules):
            if module is None:
                continue

            train_method = getattr(module, "train", None)
            if callable(train_method):
                check_missing_args(args, train_method, module)
                module.train(
                    n_epochs=n_epochs[i],
                    batch_size=batch_size[i],
                    **args)
            else:
                warnings.warn(
                    f"Module {module.__class__.__name__} is not trainable, or "
                    "it does not have a train method."
                )

    def evaluate(
        self,
        dataroot: Union[str, Path],
        dataset: str = "nuscenes",
        work_dir: Union[str, Path] = None,
        end_to_end: bool = False,
        split="val",
        **kwargs
    ):
        """Evaluate the pipeline's modules either individually, or end to end.

        :param dataroot: Root path to data dir.
        :param dataset: Dataset to evaluate on.
        :param work_dir: Path to output dir.
        :param end_to_end: Whether to evaluate the pipeline end to end.
        :param split: Which split of the dataset to evaluate on.

        Note: Model-specific arguments can also be given. 
        (e.g. map_dataroot for LaneGCN)
        """
        if work_dir is None:
            work_dir = "results/evaluation/"
            Path(work_dir).mkdir(exist_ok=True, parents=True)

        args = locals()
        del args['self']

        if end_to_end:
            if not all(self.modules):
                print(
                    "Cannot perform end-to-end evaluation if a module is "
                    "missing from the pipeline."
                )
                exit()

            _, detection_results = self.modules[0].evaluate(**args)
            _ = format_for_tracking(
                detection_results, work_dir=work_dir, save=True)

            tracking_results = self.modules[1].forward(
                dataroot=dataroot, det_path=Path(work_dir) /
                "detections_track_format.json", work_dir=work_dir)

            if "map_dataroot" not in kwargs:
                print("Missing argument 'map_dataroot' in evaluate.")
                exit()

            prediction_results = self.modules[2].forward(
                dataroot=dataroot,
                map_dataroot=kwargs["map_dataroot"],
                tracking_results=tracking_results)

            pred_results_cpu = cpu(prediction_results)

            predictions_dir = Path(work_dir) / "prediction/"
            predictions_dir.mkdir(exist_ok=True, parents=True)
            predictions_path = predictions_dir / "predictions.json"
            with open(predictions_path, "w") as f:
                json.dump(pred_results_cpu, f)

            save_path = Path(work_dir) / "prediction/"

            evaluator = Evaluator(
                task="prediction",
                dataset=dataset,
                results=predictions_path,
                output=save_path,
                dataroot=dataroot,
                split=split,
                version="v1.0-trainval",
                config_path=None,
                verbose=True,
                render_classes=None,
                render_curves=False,
                plot_examples=0)

            return evaluator.evaluate()

        for module in self.modules:
            if module is None:
                continue

            eval_method = getattr(module, "evaluate", None)
            if callable(eval_method):
                check_missing_args(args, eval_method, module)
                module.evaluate(**args)
            else:
                warnings.warn(
                    f"Module {module.__class__.__name__} does not have an "
                    "evaluate method."
                )

    def forward(self):
        pass

    def visualize(self):
        pass
