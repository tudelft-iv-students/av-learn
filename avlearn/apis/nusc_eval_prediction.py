import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from nuscenes import NuScenes
from nuscenes.eval.prediction.compute_metrics import compute_metrics
from nuscenes.eval.prediction.config import (PredictionConfig,
                                             load_prediction_config)
from nuscenes.prediction import PredictHelper


class PredictionEval:
    """
    This is unofficial nuScenes prediction evaluation code.
    Results are written to the provided output_dir.
    
    nuScenes uses the following prediction metrics:
    - Minimum Average Displacement Error over k (minADE_k): The average of pointwise L2 distances 
      between the predicted trajectory and ground truth over the k most likely predictions.
    - Minimum Final Displacement Error over k (minFDE_k): The final displacement error (FDE) is 
      the L2 distance between the final points of the prediction and ground truth. We take the minimum
      FDE over the k most likely predictions and average over all agents.
    - Miss Rate At 2 meters over k (MissRate_2_k): If the maximum pointwise L2 distance between the 
      prediction and ground truth is greater than 2 meters, we define the prediction as a miss. 
      For each agent, we take the k most likely predictions and evaluate if any are misses. 
      The MissRate_2_k is the proportion of misses over all agents.
    
    Here is an overview of the functions in this method:
    - init: Loads predictions stored in JSON format. Also loads objects necessary to compute the metrics.
    - evaluate: Performs evaluations and dumps the metric data to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.
    Please see https://www.nuscenes.org/prediction for more details.
    """
    def __init__(self, nusc: NuScenes, config: Union[str, PredictionConfig], 
                 result_path: Union[str, Path], output_dir: Union[str, Path]):
        
        if not isinstance(nusc, NuScenes):
            raise TypeError("nusc must be a NuScenes object, "
                            f"but is {type(nusc)}")
        self.helper = PredictHelper(nusc)
        
        if isinstance(config, str):
            config = load_prediction_config(self.helper, config)
        elif not isinstance(config, PredictionConfig):
            raise TypeError('config must be a string or PredictionConfig object, '
                            f'but is {type(config)}')
        self.config = config
        
        if not isinstance(result_path, Path):
            self.result_path = Path(result_path)
        if not self.result_path.is_file():
            raise FileNotFoundError(f"File not found: {result_path}")
        if not isinstance(output_dir, Path):
            self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            self.output_dir.mkdir()
        filename = str(self.result_path.stem) + '_metrics.json'
        self.submission_path = self.output_dir / filename
        
        self.predictions = json.load(open(self.result_path, 'r'))
        
    def evaluate(self, dump: bool = True) -> Dict[str, Any]:
        """
        Performs the actual evaluation.  
        :return: The raw metric data.
        """
        metrics: Dict[str, Dict[str, List[float]]] = compute_metrics(self.predictions, self.helper, self.config)
        if dump:
            json.dump(metrics, open(self.submission_path, 'w+'), indent=2)
        return metrics
    
    def main(self):
        self.evaluate()    
