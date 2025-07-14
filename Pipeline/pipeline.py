"""Pipeline class for composing transformation steps."""

import json
import os
from typing import List, Union, Optional, Dict, Any

# Import the protocol - adjust path as needed
from PipelineStep.transform_step import TransformationStep, InputData, OutputData, FeatureMatrix


class PipelinePersistence:
    """Simple persistence for Pipeline metadata using local JSON files."""
    
    def __init__(self, base_path: str = "pipelines"):
        """
        Initialize pipeline persistence.
        
        Args:
            base_path: Base directory for storing pipeline metadata
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
    
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> None:
        """Save pipeline metadata to JSON file."""
        full_path = os.path.join(self.base_path, f"{path}.json")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load pipeline metadata from JSON file."""
        full_path = os.path.join(self.base_path, f"{path}.json")
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Pipeline metadata not found at: {full_path}")
        
        with open(full_path, 'r') as f:
            return json.load(f)


class Pipeline:
    """
    A pipeline that composes multiple transformation steps together.
    
    The pipeline applies steps sequentially:
    - First step receives the original input
    - Each subsequent step receives the output of the previous step
    - Pipeline itself implements TransformationStep protocol for composability
    """
    
    def __init__(self, steps: List[TransformationStep], persistence: Optional[PipelinePersistence] = None):
        """
        Initialize the pipeline with a list of transformation steps.
        
        Args:
            steps: List of objects implementing TransformationStep protocol
                  (e.g., FeatureExtractor instances, SupervisedModel instances)
            persistence: Pipeline persistence handler for saving/loading
        
        Raises:
            ValueError: If steps list is empty
        """
        if not steps:
            raise ValueError("Pipeline must have at least one step")
        
        self.steps = steps
        self.persistence = persistence or PipelinePersistence()
    
    def fit_transform(self, X_train: InputData, X_test: Optional[InputData] = None,
                     y_train: Optional[Any] = None, y_test: Optional[Any] = None) -> OutputData:
        """
        Fit the pipeline and transform both train and test sets.
        
        Each step is fitted using fit_transform in sequence, with each step receiving
        the transformed output from the previous step.
        
        Args:
            X_train: Training data 
            X_test: Test data (optional)
            y_train: Training target labels (optional, for supervised learning steps)
            y_test: Test target labels (optional, for supervised learning steps)
            
        Returns:
            OutputData: Transformed features from the final step
        """
        current_train = X_train
        current_test = X_test
        
        # Apply each step using fit_transform in sequence
        for step in self.steps:
            if current_test is not None:
                # If we have test data, fit_transform should return a tuple
                # Try to call with target parameters if the step supports them
                try:
                    result = step.fit_transform(current_train, current_test, y_train, y_test)  # type: ignore[call-arg]
                except TypeError:
                    # Fallback for steps that don't support target parameters
                    result = step.fit_transform(current_train, current_test)  # type: ignore[call-arg]
                    
                if isinstance(result, tuple) and len(result) == 2:
                    current_train, current_test = result
                else:
                    # Handle case where step doesn't return tuple despite X_test being provided
                    current_train = result
                    current_test = step.transform(current_test) if current_test is not None else None
            else:
                # No test data, just transform training data
                # Try to call with target parameters if the step supports them
                try:
                    result = step.fit_transform(current_train, None, y_train, None)  # type: ignore[call-arg]
                except TypeError:
                    # Fallback for steps that don't support target parameters
                    result = step.fit_transform(current_train)  # type: ignore[call-arg]
                    
                if isinstance(result, tuple):
                    # In case fit_transform returns tuple even without X_test
                    current_train = result[0]
                    current_test = None
                else:
                    current_train = result
                    current_test = None
        
        # Return appropriate format
        if current_test is not None:
            return current_train, current_test  # type: ignore[return-value]
        else:
            return current_train  # type: ignore[return-value]
    
    def transform(self, X: InputData) -> FeatureMatrix:
        """
        Transform new data through the entire pipeline.
        
        Args:
            X: Input data to transform
            
        Returns:
            FeatureMatrix: Transformed features from the final step
        """
        current: Union[InputData, FeatureMatrix] = X
        for step in self.steps:
            current = step.transform(current)
        # After transformation through all steps, current should be FeatureMatrix
        return current  # type: ignore[return-value]
    
    def save(self, path: str) -> None:
        """
        Save the pipeline by saving each step individually and creating reconstruction metadata.
        
        Args:
            path: Base path for saving the pipeline
            
        Raises:
            ValueError: If any step lacks a save method or persistence object
        """
        # Pre-validate that all steps can be saved
        validation_errors = []
        for i, step in enumerate(self.steps):
            step_class_name = step.__class__.__name__
            
            # Check if step has save method
            if not (hasattr(step, 'save') and callable(getattr(step, 'save'))):
                validation_errors.append(f"Step {i} ({step_class_name}) lacks a save method")
            
            # Check if step has persistence object
            if not (hasattr(step, 'persistence') and step.persistence is not None):  # type: ignore[attr-defined]
                validation_errors.append(f"Step {i} ({step_class_name}) lacks a persistence object")
        
        # Fail early if any validation errors
        if validation_errors:
            error_msg = "Cannot save pipeline due to the following issues:\n" + "\n".join(validation_errors)
            raise ValueError(error_msg)
        
        pipeline_metadata = {
            'version': '1.0',
            'num_steps': len(self.steps),
            'steps': []
        }
        
        # Save each step and collect metadata
        for i, step in enumerate(self.steps):
            step_path = f"{path}_step_{i}"
            step_metadata = {
                'index': i,
                'class_name': step.__class__.__name__,
                'module_name': step.__class__.__module__,
                'save_path': step_path
            }
            
            # Capture persistence information (we've already validated it exists)
            persistence_obj = step.persistence  # type: ignore[attr-defined]
            persistence_metadata = {
                'has_persistence': True,
                'persistence_class': persistence_obj.__class__.__name__,
                'persistence_module': persistence_obj.__class__.__module__,
                'persistence_config': {}
            }
            
            # Capture common persistence configuration attributes
            for attr in ['bucket_name', 'base_path', 'prefix', 'region']:
                if hasattr(persistence_obj, attr):
                    persistence_metadata['persistence_config'][attr] = getattr(persistence_obj, attr)
            
            step_metadata['persistence'] = persistence_metadata
            
            # Save the step (we've already validated it has a save method)
            try:
                step.save(step_path)  # type: ignore[attr-defined]
                step_metadata['save_successful'] = True
                print(f"Saved step {i}: {step.__class__.__name__}")
            except Exception as e:
                step_metadata['save_successful'] = False
                step_metadata['save_error'] = str(e)
                raise ValueError(f"Failed to save step {i} ({step.__class__.__name__}): {e}") from e
            
            pipeline_metadata['steps'].append(step_metadata)
        
        # Save pipeline metadata
        self.persistence.save_metadata(pipeline_metadata, path)
        print(f"Pipeline metadata saved: {path}")
    
    def load(self, path: str) -> None:
        """
        Load the pipeline by reconstructing each step from metadata.
        
        Args:
            path: Base path for loading the pipeline
        """
        # Load pipeline metadata
        pipeline_metadata = self.persistence.load_metadata(path)
        
        # Reconstruct steps
        loaded_steps = []
        for step_metadata in pipeline_metadata['steps']:
            if step_metadata.get('save_successful', False):
                try:
                    # Dynamically import the step class
                    import importlib
                    step_module = importlib.import_module(step_metadata['module_name'])
                    step_class = getattr(step_module, step_metadata['class_name'])
                    
                    # Try to use load_from_path if available
                    if hasattr(step_class, 'load_from_path') and callable(getattr(step_class, 'load_from_path')):
                        # For classes like PipelineStep that have class method load_from_path
                        # Check if we need to reconstruct persistence for this step
                        persistence_info = step_metadata.get('persistence', {})
                        if persistence_info.get('has_persistence', False):
                            # Reconstruct the persistence object
                            persistence_module = importlib.import_module(persistence_info['persistence_module'])
                            persistence_class = getattr(persistence_module, persistence_info['persistence_class'])
                            persistence_config = persistence_info.get('persistence_config', {})
                            
                            # Create persistence instance with saved configuration
                            if persistence_config:
                                # Filter out None values and create persistence with available config
                                filtered_config = {k: v for k, v in persistence_config.items() if v is not None}
                                persistence = persistence_class(**filtered_config)
                            else:
                                persistence = persistence_class()
                            
                            step = step_class.load_from_path(step_metadata['save_path'], persistence)
                        else:
                            step = step_class.load_from_path(step_metadata['save_path'])
                    else:
                        # Create instance and try to load
                        step = step_class()
                        if hasattr(step, 'load') and callable(getattr(step, 'load')):
                            step.load(step_metadata['save_path'])
                        else:
                            raise ValueError(f"Cannot load step {step_metadata['index']}: no load method available")
                    
                    loaded_steps.append(step)
                    print(f"Loaded step {step_metadata['index']}: {step_metadata['class_name']}")
                
                except Exception as e:
                    raise ValueError(f"Failed to load step {step_metadata['index']} ({step_metadata['class_name']}): {e}")
            else:
                raise ValueError(f"Cannot load step {step_metadata['index']}: was not saved successfully")
        
        # Update pipeline with loaded steps
        self.steps = loaded_steps
        print(f"Pipeline loaded with {len(self.steps)} steps")
    
    @classmethod
    def load_from_path(cls, path: str, persistence: Optional[PipelinePersistence] = None) -> 'Pipeline':
        """
        Class method to create a new Pipeline instance and load from path.
        
        Args:
            path: Path to load the pipeline from
            persistence: Pipeline persistence handler
            
        Returns:
            Pipeline: New instance with loaded pipeline
        """
        pipeline = cls([], persistence=persistence)
        pipeline.load(path)
        return pipeline
    
    def add_step(self, step: TransformationStep) -> 'Pipeline':
        """
        Add a new step to the end of the pipeline.
        
        Args:
            step: TransformationStep to add
            
        Returns:
            Pipeline: New pipeline instance with the additional step
        """
        return Pipeline(self.steps + [step], persistence=self.persistence)
    
    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)
    
    def __getitem__(self, index: int) -> TransformationStep:
        """Get a specific step by index."""
        return self.steps[index]
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"Pipeline(steps={step_names})"
