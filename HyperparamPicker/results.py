"""Results analysis and visualization for hyperparameter optimization."""

import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from ax.core import Experiment, Trial
from ax.core.data import Data
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
# Note: exp_to_df removed from ax.utils.report_utils in newer versions


@dataclass
class HyperparamResults:
    """Comprehensive results from hyperparameter optimization."""
    
    experiment: Experiment
    pareto_frontier: List[Trial] = field(default_factory=list)
    best_single_objective: Dict[str, Trial] = field(default_factory=dict)
    optimization_history: List[Trial] = field(default_factory=list)
    convergence_analysis: Dict[str, Any] = field(default_factory=dict)
    results_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Post-initialization to compute results."""
        self._compute_results()
    
    def _compute_results(self):
        """Compute optimization results and analysis."""
        # Get all completed trials
        self.optimization_history = [
            trial for trial in self.experiment.trials.values()
            if trial.status.is_completed
        ]
        
        if not self.optimization_history:
            print("No completed trials found.")
            return
        
        # Compute Pareto frontier for multi-objective optimization
        if hasattr(self.experiment.optimization_config, 'objective') and \
           hasattr(self.experiment.optimization_config.objective, 'objectives'):
            self._compute_pareto_frontier()
        
        # Find best trials for each objective
        self._compute_best_single_objective()
        
        # Compute convergence analysis
        self._compute_convergence_analysis()
    
    def _compute_pareto_frontier(self):
        """Compute Pareto frontier for multi-objective optimization."""
        try:
            # Get the experiment data
            data = self.experiment.fetch_data()
            
            if data.df.empty:
                print("No data available for Pareto frontier computation.")
                return
            
            # For simplicity, we'll identify non-dominated solutions manually
            # This is a basic implementation - Ax has more sophisticated methods
            trial_metrics = {}
            
            for trial in self.optimization_history:
                trial_data = self._get_trial_metrics(trial)
                if trial_data:
                    trial_metrics[trial.index] = trial_data
            
            # Find non-dominated solutions
            pareto_indices = self._find_pareto_optimal(trial_metrics)
            self.pareto_frontier = [
                trial for trial in self.optimization_history
                if trial.index in pareto_indices
            ]
            
        except Exception as e:
            print(f"Error computing Pareto frontier: {e}")
            self.pareto_frontier = []
    
    def _compute_best_single_objective(self):
        """Find best trial for each objective."""
        if not self.optimization_history:
            return
        
        # Get all objectives
        objectives = []
        if hasattr(self.experiment.optimization_config, 'objective'):
            if hasattr(self.experiment.optimization_config.objective, 'objectives'):
                # Multi-objective
                objectives = [
                    obj.metric.name for obj in 
                    self.experiment.optimization_config.objective.objectives
                ]
            else:
                # Single objective
                objectives = [self.experiment.optimization_config.objective.metric.name]
        
        # Find best trial for each objective
        for objective in objectives:
            best_trial = None
            best_value = None
            
            for trial in self.optimization_history:
                trial_metrics = self._get_trial_metrics(trial)
                if trial_metrics and objective in trial_metrics:
                    value = trial_metrics[objective]
                    
                    # Determine if this is better
                    is_better = False
                    if best_value is None:
                        is_better = True
                    else:
                        # Check if metric should be minimized or maximized
                        metric_obj = next(
                            (obj for obj in self.experiment.optimization_config.objective.objectives 
                             if obj.metric.name == objective),
                            None
                        )
                        if metric_obj:
                            if metric_obj.minimize:
                                is_better = value < best_value
                            else:
                                is_better = value > best_value
                        else:
                            is_better = value > best_value  # Default to maximize
                    
                    if is_better:
                        best_trial = trial
                        best_value = value
            
            if best_trial:
                self.best_single_objective[objective] = best_trial
    
    def _compute_convergence_analysis(self):
        """Compute convergence analysis."""
        if not self.optimization_history:
            return
        
        # Sort trials by index (creation order)
        sorted_trials = sorted(self.optimization_history, key=lambda t: t.index)
        
        # Track best values over time for each objective
        convergence_data = {}
        objectives = list(self.best_single_objective.keys())
        
        for objective in objectives:
            best_values = []
            current_best = None
            
            for trial in sorted_trials:
                trial_metrics = self._get_trial_metrics(trial)
                if trial_metrics and objective in trial_metrics:
                    value = trial_metrics[objective]
                    
                    if current_best is None:
                        current_best = value
                    else:
                        # Update best value
                        metric_obj = next(
                            (obj for obj in self.experiment.optimization_config.objective.objectives 
                             if obj.metric.name == objective),
                            None
                        )
                        if metric_obj and metric_obj.minimize:
                            current_best = min(current_best, value)
                        else:
                            current_best = max(current_best, value)
                    
                    best_values.append(current_best)
                else:
                    best_values.append(current_best)
            
            convergence_data[objective] = best_values
        
        self.convergence_analysis = convergence_data
    
    def _get_trial_metrics(self, trial: Trial) -> Optional[Dict[str, float]]:
        """Get metric values for a trial."""
        if not self.results_dir:
            return None
        
        results_file = self.results_dir / f"trial_{trial.index}_results.json"
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                trial_data = json.load(f)
            
            results = trial_data.get("results", {})
            
            # Extract common metrics
            metrics = {}
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            
            return metrics
            
        except Exception as e:
            print(f"Error reading trial {trial.index} results: {e}")
            return None
    
    def _find_pareto_optimal(self, trial_metrics: Dict[int, Dict[str, float]]) -> List[int]:
        """Find Pareto optimal solutions."""
        if not trial_metrics:
            return []
        
        # Get objective directions (minimize/maximize)
        objective_directions = {}
        if hasattr(self.experiment.optimization_config.objective, 'objectives'):
            for obj in self.experiment.optimization_config.objective.objectives:
                objective_directions[obj.metric.name] = obj.minimize
        
        pareto_indices = []
        trial_indices = list(trial_metrics.keys())
        
        for i, trial_idx in enumerate(trial_indices):
            is_pareto_optimal = True
            trial_values = trial_metrics[trial_idx]
            
            for j, other_trial_idx in enumerate(trial_indices):
                if i == j:
                    continue
                
                other_values = trial_metrics[other_trial_idx]
                
                # Check if other trial dominates this trial
                dominates = True
                for metric_name in trial_values:
                    if metric_name in other_values:
                        minimize = objective_directions.get(metric_name, False)
                        
                        if minimize:
                            if other_values[metric_name] >= trial_values[metric_name]:
                                dominates = False
                                break
                        else:
                            if other_values[metric_name] <= trial_values[metric_name]:
                                dominates = False
                                break
                
                if dominates:
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_indices.append(trial_idx)
        
        return pareto_indices
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get optimization results as a pandas DataFrame."""
        if not self.optimization_history:
            return pd.DataFrame()
        
        rows = []
        
        for trial in self.optimization_history:
            row = {
                "trial_index": trial.index,
                "status": trial.status.name,
                "is_pareto_optimal": trial.index in [t.index for t in self.pareto_frontier]
            }
            
            # Add hyperparameters
            if trial.arm:
                for param_name, param_value in trial.arm.parameters.items():
                    row[f"param_{param_name}"] = param_value
            
            # Add metrics
            trial_metrics = self._get_trial_metrics(trial)
            if trial_metrics:
                for metric_name, metric_value in trial_metrics.items():
                    row[f"metric_{metric_name}"] = metric_value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_recommended_configs(self, 
                              preference_weights: Optional[Dict[str, float]] = None) -> List[Trial]:
        """Get recommended configurations based on user preferences."""
        if not self.pareto_frontier:
            return list(self.best_single_objective.values())
        
        if preference_weights is None:
            # Return all Pareto optimal solutions
            return self.pareto_frontier
        
        # Score each Pareto optimal solution based on preferences
        scored_trials = []
        
        for trial in self.pareto_frontier:
            trial_metrics = self._get_trial_metrics(trial)
            if not trial_metrics:
                continue
            
            score = 0.0
            total_weight = 0.0
            
            for metric_name, weight in preference_weights.items():
                if metric_name in trial_metrics:
                    # Normalize score based on metric direction
                    metric_obj = next(
                        (obj for obj in self.experiment.optimization_config.objective.objectives 
                         if obj.metric.name == metric_name),
                        None
                    )
                    
                    value = trial_metrics[metric_name]
                    if metric_obj and metric_obj.minimize:
                        # For minimization, lower values get higher scores
                        normalized_score = 1.0 / (1.0 + value)
                    else:
                        # For maximization, higher values get higher scores
                        normalized_score = value
                    
                    score += weight * normalized_score
                    total_weight += weight
            
            if total_weight > 0:
                score /= total_weight
                scored_trials.append((trial, score))
        
        # Sort by score (descending)
        scored_trials.sort(key=lambda x: x[1], reverse=True)
        
        return [trial for trial, score in scored_trials]
    
    def export_results(self, filepath: str):
        """Export results to JSON file."""
        results_data = {
            "experiment_name": self.experiment.name,
            "total_trials": len(self.optimization_history),
            "pareto_frontier_size": len(self.pareto_frontier),
            "best_single_objective": {
                metric: trial.index for metric, trial in self.best_single_objective.items()
            },
            "convergence_analysis": self.convergence_analysis,
            "results_dataframe": self.get_results_dataframe().to_dict(orient="records")
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results exported to {filepath}")
    
    def print_summary(self):
        """Print a summary of optimization results."""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"Total trials completed: {len(self.optimization_history)}")
        print(f"Pareto frontier size: {len(self.pareto_frontier)}")
        
        print("\nBest single objective results:")
        for metric, trial in self.best_single_objective.items():
            trial_metrics = self._get_trial_metrics(trial)
            if trial_metrics and metric in trial_metrics:
                print(f"  {metric}: {trial_metrics[metric]:.4f} (Trial {trial.index})")
        
        if self.pareto_frontier:
            print(f"\nPareto optimal trials: {[t.index for t in self.pareto_frontier]}")
        
        print("\n" + "="*60) 