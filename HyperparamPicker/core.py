"""Main HyperparamPicker class for hyperparameter optimization."""

import tempfile
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ax.core import Experiment, SearchSpace, Arm, Trial
from ax.core.optimization_config import OptimizationConfig
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.generation_strategy.dispatch_utils import choose_generation_strategy_legacy as choose_generation_strategy

from DataLoader import DataSourceType
from FeatureExtractor import FeatureExtractorType
from SupervisedModel import SupervisedModelType

from .config import HyperparamSearchConfig, MultiObjectiveConfig
from .runner import PipelineRunner
from .metrics import create_metrics_for_objectives
from .results import HyperparamResults


class HyperparamPicker:
    """Main class for hyperparameter optimization using Facebook AI Research's Ax."""
    
    def __init__(self,
                 search_config: HyperparamSearchConfig,
                 multi_objective_config: MultiObjectiveConfig,
                 experiment_name: str = "hyperparameter_optimization"):
        """
        Initialize the HyperparamPicker.
        
        Args:
            search_config: Configuration for hyperparameter search space
            multi_objective_config: Configuration for multi-objective optimization
            experiment_name: Name for the experiment
        """
        self.search_config = search_config
        self.multi_objective_config = multi_objective_config
        self.experiment_name = experiment_name
        
        # Will be set during optimization
        self.experiment: Optional[Experiment] = None
        self.scheduler: Optional[Scheduler] = None
        self.runner: Optional[PipelineRunner] = None
        self.results_dir: Optional[Path] = None
    
    def optimize(self,
                 data_source_type: DataSourceType,
                 feature_extractor_type: FeatureExtractorType,
                 model_type: SupervisedModelType,
                 loader_kwargs: Dict[str, Any],
                 base_extractor_kwargs: Dict[str, Any],
                 base_model_kwargs: Dict[str, Any],
                 total_trials: int = 50,
                 max_parallel_trials: int = 4,
                 use_class_weights: bool = True,
                 results_dir: Optional[str] = None) -> HyperparamResults:
        """
        Run hyperparameter optimization.
        
        Args:
            data_source_type: Type of data source
            feature_extractor_type: Type of feature extractor
            model_type: Type of model
            loader_kwargs: Arguments for data loader
            base_extractor_kwargs: Base arguments for feature extractor
            base_model_kwargs: Base arguments for model
            total_trials: Total number of trials to run
            max_parallel_trials: Maximum number of parallel trials
            use_class_weights: Whether to use class weights
            results_dir: Directory to store results
            
        Returns:
            HyperparamResults object containing optimization results
        """
        print(f"Starting hyperparameter optimization with {total_trials} trials")
        print(f"Feature extractor: {feature_extractor_type}")
        print(f"Model: {model_type}")
        
        # Set up results directory
        if results_dir is None:
            self.results_dir = Path(tempfile.mkdtemp(prefix="hyperparam_"))
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create search space
        search_space = self._create_search_space(feature_extractor_type, model_type)
        
        # Create metrics
        objective_names = [obj[0] for obj in self.multi_objective_config.objectives]
        metrics = create_metrics_for_objectives(objective_names, str(self.results_dir))
        
        # Create optimization config
        optimization_config = self.multi_objective_config.to_ax_optimization_config(metrics)
        
        # Create runner
        self.runner = PipelineRunner(
            data_source_type=data_source_type,
            feature_extractor_type=feature_extractor_type,
            model_type=model_type,
            loader_kwargs=loader_kwargs,
            base_extractor_kwargs=base_extractor_kwargs,
            base_model_kwargs=base_model_kwargs,
            use_class_weights=use_class_weights,
            results_dir=str(self.results_dir)
        )
        
        # Create experiment
        self.experiment = Experiment(
            name=self.experiment_name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=self.runner
        )
        
        # Choose generation strategy
        generation_strategy = choose_generation_strategy(
            search_space=search_space,
            optimization_config=optimization_config,
            num_trials=total_trials
        )
        
        # Create scheduler
        self.scheduler = Scheduler(
            experiment=self.experiment,
            generation_strategy=generation_strategy,
            options=SchedulerOptions(
                total_trials=total_trials,
                max_pending_trials=max_parallel_trials
            )
        )
        
        # Run optimization
        print("Running optimization...")
        self.scheduler.run_all_trials()
        
        print(f"Optimization completed! Results saved to {self.results_dir}")
        
        # Create and return results
        results = HyperparamResults(
            experiment=self.experiment,
            results_dir=self.results_dir
        )
        
        results.print_summary()
        
        return results
    
    def _create_search_space(self,
                           feature_extractor_type: FeatureExtractorType,
                           model_type: SupervisedModelType) -> SearchSpace:
        """Create Ax search space from configuration."""
        # Get parameters for this specific configuration
        parameters = self.search_config.get_parameters_for_config(
            feature_extractor_type, model_type
        )
        
        # Convert to Ax parameters
        ax_parameters = []
        for param_spec in parameters:
            try:
                ax_param = param_spec.to_ax_parameter()
                ax_parameters.append(ax_param)
            except Exception as e:
                print(f"Error creating parameter {param_spec.name}: {e}")
                continue
        
        if not ax_parameters:
            raise ValueError("No valid parameters found for search space")
        
        return SearchSpace(parameters=ax_parameters)
    
    def get_best_trial(self, metric_name: str) -> Optional[Trial]:
        """
        Get the best trial for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Best trial or None if not found
        """
        if not self.experiment:
            return None
        
        best_trial = None
        best_value = None
        
        for trial in self.experiment.trials.values():
            if not trial.status.is_completed:
                continue
            
            # Get trial results
            results_file = self.results_dir / f"trial_{trial.index}_results.json"
            if not results_file.exists():
                continue
            
            try:
                import json
                with open(results_file, 'r') as f:
                    trial_data = json.load(f)
                
                results = trial_data.get("results", {})
                if metric_name in results:
                    value = results[metric_name]
                    
                    if best_value is None or value > best_value:  # Assuming higher is better
                        best_trial = trial
                        best_value = value
                        
            except Exception as e:
                print(f"Error reading trial {trial.index}: {e}")
                continue
        
        return best_trial
    
    def get_trial_results(self, trial_index: int) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific trial.
        
        Args:
            trial_index: Index of the trial
            
        Returns:
            Trial results or None if not found
        """
        if not self.results_dir:
            return None
        
        results_file = self.results_dir / f"trial_{trial_index}_results.json"
        if not results_file.exists():
            return None
        
        try:
            import json
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading trial {trial_index}: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.runner:
            self.runner.cleanup()
        
        # Note: We don't delete results_dir as it contains valuable results 