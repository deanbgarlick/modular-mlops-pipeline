"""Feature extractor persistence interfaces and implementations."""

import pickle
import io
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import storage


class FeatureExtractorPersistence(ABC):
    """Abstract base class for feature extractor persistence."""
    
    @abstractmethod
    def save(self, extractor: Any, path: str) -> None:
        """
        Save a feature extractor to the specified path.
        
        Args:
            extractor: The feature extractor object to save
            path: The path where to save the extractor
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Load a feature extractor from the specified path.
        
        Args:
            path: The path to load the extractor from
            
        Returns:
            The loaded feature extractor object
        """
        pass


class PickleGCPExtractorPersistence(FeatureExtractorPersistence):
    """GCP bucket persistence for feature extractors using pickle serialization."""
    
    def __init__(self, bucket_name: str, prefix: str = "feature_extractors/"):
        """
        Initialize GCP bucket persistence with pickle serialization.
        
        Uses environment variables for authentication:
        - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file (optional)
        - Or uses default credentials (gcloud auth, service account, etc.)
        
        Args:
            bucket_name: Name of the GCP bucket
            prefix: Prefix path within bucket for feature extractors
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
    
    def save(self, extractor: Any, path: str) -> None:
        """Save feature extractor to GCP bucket using pickle."""
        client, bucket = self._get_client()
        
        # Add prefix to path
        full_path = f"{self.prefix}{path}"
        
        # Serialize extractor to bytes using pickle
        buffer = io.BytesIO()
        pickle.dump(extractor, buffer)
        buffer.seek(0)
        
        # Upload to GCP bucket
        blob = bucket.blob(full_path)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        print(f"Feature extractor saved to GCP bucket: gs://{self.bucket_name}/{full_path}")
    
    def load(self, path: str) -> Any:
        """Load feature extractor from GCP bucket using pickle."""
        client, bucket = self._get_client()
        
        # Add prefix to path
        full_path = f"{self.prefix}{path}"
        
        # Download from GCP bucket
        blob = bucket.blob(full_path)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        
        # Deserialize extractor using pickle
        extractor = pickle.load(buffer)
        print(f"Feature extractor loaded from GCP bucket: gs://{self.bucket_name}/{full_path}")
        return extractor


class PickleAWSExtractorPersistence(FeatureExtractorPersistence):
    """AWS S3 bucket persistence for feature extractors using pickle serialization."""
    
    def __init__(self, bucket_name: str, prefix: str = "feature_extractors/"):
        """
        Initialize AWS S3 persistence with pickle serialization.
        
        Uses environment variables for configuration:
        - AWS_DEFAULT_REGION or AWS_REGION: AWS region (defaults to 'us-east-1')
        - AWS_ACCESS_KEY_ID: AWS access key ID (optional, can use IAM roles)
        - AWS_SECRET_ACCESS_KEY: AWS secret access key (optional, can use IAM roles)
        - AWS_PROFILE: AWS profile name (optional)
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: Prefix path within bucket for feature extractors
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
    
    def save(self, extractor: Any, path: str) -> None:
        """Save feature extractor to AWS S3 using pickle."""
        client = self._get_client()
        
        # Add prefix to path
        full_path = f"{self.prefix}{path}"
        
        # Serialize extractor to bytes using pickle
        buffer = io.BytesIO()
        pickle.dump(extractor, buffer)
        buffer.seek(0)
        
        # Upload to S3
        client.upload_fileobj(buffer, self.bucket_name, full_path)
        print(f"Feature extractor saved to AWS S3: s3://{self.bucket_name}/{full_path}")
    
    def load(self, path: str) -> Any:
        """Load feature extractor from AWS S3 using pickle."""
        client = self._get_client()
        
        # Add prefix to path
        full_path = f"{self.prefix}{path}"
        
        # Download from S3
        buffer = io.BytesIO()
        client.download_fileobj(self.bucket_name, full_path, buffer)
        buffer.seek(0)
        
        # Deserialize extractor using pickle
        extractor = pickle.load(buffer)
        print(f"Feature extractor loaded from AWS S3: s3://{self.bucket_name}/{full_path}")
        return extractor


class PickleLocalExtractorPersistence(FeatureExtractorPersistence):
    """Local file system persistence for feature extractors using pickle serialization."""
    
    def __init__(self, base_path: str = "feature_extractors"):
        """
        Initialize local file persistence with pickle serialization.
        
        Args:
            base_path: Base directory for storing feature extractors
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
    
    def save(self, extractor: Any, path: str) -> None:
        """Save feature extractor to local file system using pickle."""
        full_path = os.path.join(self.base_path, path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Save extractor using pickle
        with open(full_path, 'wb') as f:
            pickle.dump(extractor, f)
        
        print(f"Feature extractor saved locally: {full_path}")
    
    def load(self, path: str) -> Any:
        """Load feature extractor from local file system using pickle."""
        full_path = os.path.join(self.base_path, path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Feature extractor not found at: {full_path}")
        
        with open(full_path, 'rb') as f:
            extractor = pickle.load(f)
        
        print(f"Feature extractor loaded locally: {full_path}")
        return extractor


class HuggingFaceExtractorPersistence(FeatureExtractorPersistence):
    """Specialized persistence for HuggingFace transformers with proper model handling."""
    
    def __init__(self, bucket_name: Optional[str] = None, base_path: str = "feature_extractors", 
                 use_gcp: bool = True):
        """
        Initialize HuggingFace transformer persistence.
        
        Args:
            bucket_name: GCP/AWS bucket name (if using cloud storage)
            base_path: Local base directory for storing extractors
            use_gcp: Whether to use GCP storage (True) or local storage (False)
        """
        self.use_gcp = use_gcp and bucket_name is not None
        
        if self.use_gcp and bucket_name is not None:
            self.gcp_persistence = PickleGCPExtractorPersistence(
                bucket_name, prefix="huggingface_extractors/"
            )
        else:
            self.local_persistence = PickleLocalExtractorPersistence(
                os.path.join(base_path, "huggingface")
            )
    
    def save(self, extractor: Any, path: str) -> None:
        """Save HuggingFace extractor with proper handling."""
        # For HuggingFace extractors, we typically want to save the model state
        # and configuration separately for better compatibility
        
        extractor_data = {
            'extractor_type': extractor.__class__.__name__,
            'extractor_state': extractor,
            'feature_info': extractor.get_feature_info() if hasattr(extractor, 'get_feature_info') else {}
        }
        
        if self.use_gcp:
            self.gcp_persistence.save(extractor_data, path)
        else:
            self.local_persistence.save(extractor_data, path)
    
    def load(self, path: str) -> Any:
        """Load HuggingFace extractor with proper handling."""
        if self.use_gcp:
            extractor_data = self.gcp_persistence.load(path)
        else:
            extractor_data = self.local_persistence.load(path)
        
        if isinstance(extractor_data, dict) and 'extractor_state' in extractor_data:
            return extractor_data['extractor_state']
        else:
            # Backward compatibility - direct extractor object
            return extractor_data


# Backward compatibility aliases
GCPExtractorPersistence = PickleGCPExtractorPersistence
AWSExtractorPersistence = PickleAWSExtractorPersistence
LocalExtractorPersistence = PickleLocalExtractorPersistence 