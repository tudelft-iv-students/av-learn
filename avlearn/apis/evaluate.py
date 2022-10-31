"""This module implements the evaluation of each individual task in the
AV-learn pipeline."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.prediction.compute_metrics import compute_metrics
from nuscenes.eval.prediction.config import (PredictionConfig,
                                             load_prediction_config)
from nuscenes.eval.tracking.data_classes import TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.prediction import PredictHelper


class Evaluator:
    """Evaluate a `task` of the AV-learn pipeline.

    This class implements the evaluation of the specified `task` for the
    corresponding `dataset`. The results are written to the given `output`
    directory. 

    In the case of NuScenes, it utilizes the official evaluation code.
    https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval
    """

    def __init__(self,
                 task: Union[str, Path],
                 dataset: Union[str, Path],
                 results: Union[str, Path],
                 output: Union[str, Path],
                 dataroot: Union[str, Path],
                 split: Union[str, Path],
                 verbose: bool = True,
                 **kwargs: Any
                 ) -> None:
        """Initialize an Evaluator.

        :param task: Task to evaluate.
        :param dataset: The dataset to evaluate on.
        :param results: Path to the JSON file with the results of the task.
        :param output: Path to the directory to save results to.
        :param dataroot: Path to the data directory.
        :param split: The dataset split to evaluate on.
        :param verbose: Whether to print to stdout.


        Optional parameters for nuScenes dataset:
        :param version (str): Which version of nuscenes to use 
                              (Default: v1.0-trainval)
        :param config_path (str | Path): Path to the configuration file for 
                                         the evaluation. (Default: None)
        :param plot_examples (int): How many example visualizations to write 
                                    to disk. (Default: 10)
        :param render_curves (bool): Whether to render curves to disk.
                                     (Default: True)
        :param render_classes (list): For which classes to render tracking 
                                      results to disk. (Default: None -> all)

        """

        self.task = task
        self.dataset = dataset
        self.results = results
        self.output = output
        self.dataroot = dataroot
        self.split = split
        self.kwargs = kwargs

        if dataset == 'nuscenes':
            self.__init_nuscenes()
        else:
            raise NotImplementedError(
                "We currently only support the NuScenes dataset.")

    def __init_nuscenes(self) -> None:
        """Initialize a NuScenes Evaluator."""
        if self.task == "detection":
            # Initialize NuScenes object
            nusc = NuScenes(
                version=self.kwargs.get("version", 'v1.0-trainval'),
                verbose=self.kwargs.get("verbose", True),
                dataroot=self.dataroot)

            if self.kwargs.get('config_path', None) is None:
                config = config_factory('detection_cvpr_2019')
            else:
                with open(self.kwargs['config_path'], 'r') as f:
                    config = DetectionConfig.deserialize(json.load(f))

            self.evaluator = DetectionEval(
                nusc, config=config, result_path=self.results,
                eval_set=self.split, output_dir=self.output,
                verbose=self.kwargs.get("verbose", True))

        elif self.task == 'tracking':
            if self.kwargs.get('config_path', None) is None:
                config = config_factory('tracking_nips_2019')
            else:
                with open(self.kwargs['config_path'], 'r') as f:
                    config = TrackingConfig.deserialize(json.load(f))

            self.evaluator = TrackingEval(
                config=config, result_path=self.results, eval_set=self.split,
                output_dir=self.output, nusc_version=self.kwargs.get(
                    "version", "v1.0-trainval"),
                nusc_dataroot=self.dataroot, verbose=self.kwargs.get(
                    "verbose", True),
                render_classes=self.kwargs.get('render_classes', []))

        else:
            nusc = NuScenes(
                version=self.kwargs.get("version", 'v1.0-trainval'),
                verbose=self.kwargs.get("verbose", True),
                dataroot=self.dataroot)
            
            self.helper = PredictHelper(nusc)
            
            config_name = self.kwargs.get('config_name', 'predict_2020_icra.json')
            self.config: PredictionConfig = load_prediction_config(self.helper, config_name)
            
            if not isinstance(self.results, Path):
                self.results = Path(self.results)
            if not self.results.is_file():
                raise FileNotFoundError(f"File not found: {self.results}")
            if not isinstance(self.output, Path):
                self.output = Path(self.output)
            if not self.output.is_dir():
                self.output.mkdir()
            
            filename = str(self.results.stem) + '_metrics.json'
            self.submission_path = self.output / filename
            
            self.predictions = json.load(open(self.results, 'r'))

    def __init_kitti(self) -> None:
        """Initialize a KITTI Evaluator."""
        # TODO: Add support for KITTI dataset
        pass

    def __init_waymo(self) -> None:
        """Initialize a Waymo Evaluator."""
        # TODO: Add support for Waymo dataset
        pass

    def __eval_nuscenes(self) -> Dict[str, Any]:
        """Evaluate on NuScenes."""
        if self.task == 'detection':
            return self.evaluator.main(
                plot_examples=self.kwargs.get("plot_examples", 10),
                render_curves=self.kwargs.get("render_curves", True))
        elif self.task == 'tracking':
            return self.evaluator.main(
                render_curves=self.kwargs.get("render_curves", True))
        else:
            metrics: Dict[str, Dict[str, List[float]]] = compute_metrics(self.predictions, self.helper, self.config)
            json.dump(metrics, open(self.submission_path, 'w+'), indent=2)
            return metrics

    def __eval_kitti(self) -> Dict[str, Any]:
        """Evaluate on KITTI."""
        # TODO: Add support for KITTI dataset
        pass

    def __eval_waymo(self) -> Dict[str, Any]:
        """Evaluate on Waymo."""
        # TODO: Add support for Waymo dataset
        pass

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the `task`.

        This method evaluates the `task` on the given `dataset`, and renders
        corresponding plots. It returns the metrics computed during the 
        evaluation. 
        """
        if self.dataset == 'nuscenes':
            return self.__eval_nuscenes()
        elif self.dataset == 'kitti':
            pass
        else:  # Waymo dataset
            pass


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Evaluate the results.")

    # Main arguments
    parser.add_argument(
        '-t', "--task", type=str, help="Task to evaluate.",
        choices=["detection", "tracking", "prediction"], required=True)
    parser.add_argument('-r', "--results", type=str, required=True,
                        help="Path to the task\'s results.")
    parser.add_argument(
        "--dataset", type=str,
        help="Dataset to evaluate on.",
        required=True, choices=["nuscenes", "kitti", "waymo"])
    parser.add_argument(
        '-o', '--output', type=str, default='../nuscenes-metrics',
        help='Directory to save results to.')
    parser.add_argument(
        '--split', type=str, default='val',
        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str,
                        default='../data/nuscenes',
                        help='Default nuScenes data directory.')

    # nuscenes-devkit arguments
    parser.add_argument(
        '--version', type=str, default='v1.0-trainval',
        help='Which version of the dataset to evaluate on, e.g. v1.0-trainval')
    parser.add_argument(
        '--config_path', type=str, default=None,
        help='Path to the configuration file for the evaluation.')
    parser.add_argument(
        '--plot_examples', type=int, default=10,
        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render curves to disk.')
    parser.add_argument(
        '--render_classes', type=str, default=None, nargs='+',
        help='For which classes to render tracking results to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')

    args = vars(parser.parse_args())
    args['render_curves'] = bool(args['render_curves'])
    args['verbose'] = bool(args['verbose'])

    evaluator = Evaluator(**args)
    evaluator.evaluate()
