"""PipelineStep persistence interfaces and implementations."""

import json
import io
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import storage

# Import only the abstract base class for type hints
from FeatureExtractor.persistence import FeatureExtractorPersistence


class PipelineStepPersistence(ABC):
    """Abstract base class for PipelineStep persistence."""
    
    @abstractmethod
    def save(self, pipeline_step: Any, path: str) -> None:
        """
        Save a PipelineStep to the specified path.
        
        Args:
            pipeline_step: The PipelineStep object to save
            path: The path where to save the pipeline step
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Load a PipelineStep from the specified path.
        
        Args:
            path: The path to load the pipeline step from
            
        Returns:
            The loaded PipelineStep object
        """
        pass


class BasePipelineStepPersistence(PipelineStepPersistence):
    """Base class with common JSON metadata logic for PipelineStep persistence."""
    
    def _create_metadata(self, pipeline_step: Any, fe_path: str) -> Dict[str, Any]:
        """Create comprehensive JSON metadata for PipelineStep."""
        return {
            'pipeline_step_metadata': {
                'text_column_name': pipeline_step.text_column_name,
                'version': '1.0'
            },
            'feature_extractor_metadata': {
                'class_name': pipeline_step.feature_extractor.__class__.__name__,
                'module_name': pipeline_step.feature_extractor.__class__.__module__,
                'persistence_class': pipeline_step.feature_extractor.persistence.__class__.__name__,
                'persistence_module': pipeline_step.feature_extractor.persistence.__class__.__module__,
                'save_path': fe_path,  # Store the path we used to save it
                'feature_info': pipeline_step.feature_extractor.get_feature_info(),
                'persistence_config': {
                    'bucket_name': getattr(pipeline_step.feature_extractor.persistence, 'bucket_name', None),
                    'prefix': getattr(pipeline_step.feature_extractor.persistence, 'prefix', None),
                    'base_path': getattr(pipeline_step.feature_extractor.persistence, 'base_path', None),
                    'region': getattr(pipeline_step.feature_extractor.persistence, 'region', None)
                }
            }
        }
    
    def _reconstruct_feature_extractor(self, fe_metadata: Dict[str, Any]) -> Any:
        """Reconstruct FeatureExtractor from metadata."""
        # Import the FeatureExtractor class dynamically
        import importlib
        fe_module = importlib.import_module(fe_metadata['module_name'])
        fe_class = getattr(fe_module, fe_metadata['class_name'])
        
        # Import and reconstruct the FeatureExtractor's persistence
        persistence_module = importlib.import_module(fe_metadata['persistence_module'])
        persistence_class = getattr(persistence_module, fe_metadata['persistence_class'])
        
        # Reconstruct persistence using stored configuration
        persistence_config = fe_metadata.get('persistence_config', {})
        
        if 'GCP' in fe_metadata['persistence_class']:
            # For GCP persistence, use stored bucket name and prefix
            bucket_name = persistence_config.get('bucket_name')
            if not bucket_name:
                # Fallback to our own bucket if we have one
                bucket_name = getattr(self, 'bucket_name', None)
                if not bucket_name:
                    raise ValueError("Missing bucket_name in persistence config for GCP FeatureExtractor")
            fe_persistence = persistence_class(
                bucket_name=bucket_name,
                prefix=persistence_config.get('prefix', 'feature_extractors/')
            )
        elif 'AWS' in fe_metadata['persistence_class']:
            # For AWS persistence, use stored bucket name and prefix
            bucket_name = persistence_config.get('bucket_name')
            if not bucket_name:
                # Fallback to our own bucket if we have one
                bucket_name = getattr(self, 'bucket_name', None)
                if not bucket_name:
                    raise ValueError("Missing bucket_name in persistence config for AWS FeatureExtractor")
            fe_persistence = persistence_class(
                bucket_name=bucket_name,
                prefix=persistence_config.get('prefix', 'feature_extractors/')
            )
        elif 'Local' in fe_metadata['persistence_class']:
            # For local, use stored base path or default
            base_path = persistence_config.get('base_path')
            if not base_path:
                # Try to derive from our own base_path if we have one
                if hasattr(self, 'base_path'):
                    our_base_path = getattr(self, 'base_path')
                    base_path = os.path.join(os.path.dirname(our_base_path), "feature_extractors")
                else:
                    base_path = 'feature_extractors'
            fe_persistence = persistence_class(base_path=base_path)
        else:
            # Fallback for unknown persistence types
            raise ValueError(f"Unknown persistence class: {fe_metadata['persistence_class']}")
        
        # Load the FeatureExtractor with reconstructed persistence
        return fe_class.load_from_path(fe_metadata['save_path'], fe_persistence)
    
    @abstractmethod
    def _upload_json(self, metadata_json: str, path: str) -> None:
        """Upload JSON metadata to storage."""
        pass
    
    @abstractmethod
    def _download_json(self, path: str) -> str:
        """Download JSON metadata from storage."""
        pass
    
    @abstractmethod
    def _get_storage_info(self) -> str:
        """Get storage location info for logging."""
        pass
    
    def save(self, pipeline_step: Any, path: str) -> None:
        """Save PipelineStep using JSON metadata format."""
        # Define FeatureExtractor save path
        fe_path = f"{path}_feature_extractor"
        
        # Create comprehensive JSON metadata
        pipeline_step_data = self._create_metadata(pipeline_step, fe_path)
        
        # Let FeatureExtractor save itself using its own persistence system
        pipeline_step.feature_extractor.save(fe_path)
        
        # Upload JSON metadata
        metadata_json = json.dumps(pipeline_step_data, indent=2)
        self._upload_json(metadata_json, path)
        
        print(f"PipelineStep saved to {self._get_storage_info()}/{path}")
    
    def load(self, path: str) -> Any:
        """Load PipelineStep using JSON metadata format."""
        # Download JSON metadata
        metadata_json = self._download_json(path)
        pipeline_step_data = json.loads(metadata_json)
        
        # Extract metadata
        ps_metadata = pipeline_step_data['pipeline_step_metadata']
        fe_metadata = pipeline_step_data['feature_extractor_metadata']
        
        # Reconstruct FeatureExtractor
        feature_extractor = self._reconstruct_feature_extractor(fe_metadata)
        
        # Reconstruct PipelineStep
        from .pipeline_step import PipelineStep
        pipeline_step = PipelineStep(feature_extractor, ps_metadata['text_column_name'])
        
        print(f"PipelineStep loaded from {self._get_storage_info()}/{path}")
        return pipeline_step


class GCPPipelineStepPersistence(BasePipelineStepPersistence):
    """GCP bucket persistence for PipelineSteps using JSON metadata format."""
    
    def __init__(self, bucket_name: str, prefix: str = "pipeline_steps/"):
        """
        Initialize GCP bucket persistence for PipelineSteps.
        
        Uses environment variables for authentication:
        - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file (optional)
        - Or uses default credentials (gcloud auth, service account, etc.)
        
        Args:
            bucket_name: Name of the GCP bucket
            prefix: Prefix path within bucket for pipeline steps
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self._client = None
        self._bucket = None
    
    def _get_client(self) -> Tuple[Any, Any]:
        """Lazy initialization of GCP client."""
        if self._client is None:
            try:
                from google.cloud import storage  # type: ignore
                # Use default credentials from environment
                self._client = storage.Client()
                self._bucket = self._client.bucket(self.bucket_name)
            except ImportError:
                raise ImportError("google-cloud-storage package is required for GCP persistence")
        assert self._client is not None and self._bucket is not None
        return self._client, self._bucket
    
    def _upload_json(self, metadata_json: str, path: str) -> None:
        """Upload JSON metadata to GCP bucket."""
        client, bucket = self._get_client()
        full_metadata_path = f"{self.prefix}{path}.json"
        blob = bucket.blob(full_metadata_path)
        blob.upload_from_string(metadata_json, content_type='application/json')
    
    def _download_json(self, path: str) -> str:
        """Download JSON metadata from GCP bucket."""
        client, bucket = self._get_client()
        full_metadata_path = f"{self.prefix}{path}.json"
        blob = bucket.blob(full_metadata_path)
        return blob.download_as_text()
    
    def _get_storage_info(self) -> str:
        """Get GCP storage location info for logging."""
        return f"GCP bucket: gs://{self.bucket_name}/{self.prefix}"


class AWSPipelineStepPersistence(BasePipelineStepPersistence):
    """AWS S3 bucket persistence for PipelineSteps using JSON metadata format."""
    
    def __init__(self, bucket_name: str, prefix: str = "pipeline_steps/"):
        """
        Initialize AWS S3 persistence for PipelineSteps.
        
        Uses environment variables for configuration:
        - AWS_DEFAULT_REGION or AWS_REGION: AWS region (defaults to 'us-east-1')
        - AWS_ACCESS_KEY_ID: AWS access key ID (optional, can use IAM roles)
        - AWS_SECRET_ACCESS_KEY: AWS secret access key (optional, can use IAM roles)
        - AWS_PROFILE: AWS profile name (optional)
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: Prefix path within bucket for pipeline steps
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION', 'us-east-1')
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of AWS client."""
        if self._client is None:
            try:
                import boto3  # type: ignore
                # boto3 will automatically use environment variables and IAM roles
                self._client = boto3.client('s3', region_name=self.region)
            except ImportError:
                raise ImportError("boto3 package is required for AWS persistence")
        return self._client
    
    def _upload_json(self, metadata_json: str, path: str) -> None:
        """Upload JSON metadata to AWS S3."""
        client = self._get_client()
        full_metadata_path = f"{self.prefix}{path}.json"
        buffer = io.BytesIO(metadata_json.encode('utf-8'))
        client.upload_fileobj(buffer, self.bucket_name, full_metadata_path)
    
    def _download_json(self, path: str) -> str:
        """Download JSON metadata from AWS S3."""
        client = self._get_client()
        full_metadata_path = f"{self.prefix}{path}.json"
        buffer = io.BytesIO()
        client.download_fileobj(self.bucket_name, full_metadata_path, buffer)
        buffer.seek(0)
        return buffer.read().decode('utf-8')
    
    def _get_storage_info(self) -> str:
        """Get AWS storage location info for logging."""
        return f"AWS S3: s3://{self.bucket_name}/{self.prefix}"


class LocalPipelineStepPersistence(BasePipelineStepPersistence):
    """Local file system persistence for PipelineSteps using JSON metadata format."""
    
    def __init__(self, base_path: str = "pipeline_steps"):
        """
        Initialize local file persistence for PipelineSteps.
        
        Args:
            base_path: Base directory for storing pipeline steps
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
    
    def _upload_json(self, metadata_json: str, path: str) -> None:
        """Save JSON metadata to local file system."""
        full_metadata_path = os.path.join(self.base_path, f"{path}.json")
        os.makedirs(os.path.dirname(full_metadata_path), exist_ok=True)
        
        with open(full_metadata_path, 'w') as f:
            f.write(metadata_json)
    
    def _download_json(self, path: str) -> str:
        """Load JSON metadata from local file system."""
        full_metadata_path = os.path.join(self.base_path, f"{path}.json")
        
        if not os.path.exists(full_metadata_path):
            raise FileNotFoundError(f"PipelineStep metadata not found at: {full_metadata_path}")
        
        with open(full_metadata_path, 'r') as f:
            return f.read()
    
    def _get_storage_info(self) -> str:
        """Get local storage location info for logging."""
        return f"locally: {self.base_path}" 