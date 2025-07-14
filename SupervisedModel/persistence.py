"""Model persistence interfaces and implementations for supervised models."""

import pickle
import io
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from google.cloud import storage


class ModelPersistence(ABC):
    """Abstract base class for model persistence."""
    
    @abstractmethod
    def save(self, model: Any, path: str) -> None:
        """
        Save a model to the specified path.
        
        Args:
            model: The model object to save
            path: The path where to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Load a model from the specified path.
        
        Args:
            path: The path to load the model from
            
        Returns:
            The loaded model object
        """
        pass


class PickleGCPBucketPersistence(ModelPersistence):
    """GCP bucket persistence using pickle serialization."""
    
    def __init__(self, bucket_name: str):
        """
        Initialize GCP bucket persistence with pickle serialization.
        
        Uses environment variables for authentication:
        - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file (optional)
        - Or uses default credentials (gcloud auth, service account, etc.)
        
        Args:
            bucket_name: Name of the GCP bucket
        """
        self.bucket_name = bucket_name
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
    
    def save(self, model: Any, path: str) -> None:
        """Save model to GCP bucket using pickle."""
        client, bucket = self._get_client()
        
        # Serialize model to bytes using pickle
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)
        
        # Upload to GCP bucket
        blob = bucket.blob(path)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        print(f"Model saved to GCP bucket (pickle): gs://{self.bucket_name}/{path}")
    
    def load(self, path: str) -> Any:
        """Load model from GCP bucket using pickle."""
        client, bucket = self._get_client()
        
        # Download from GCP bucket
        blob = bucket.blob(path)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        
        # Deserialize model using pickle
        model = pickle.load(buffer)
        print(f"Model loaded from GCP bucket (pickle): gs://{self.bucket_name}/{path}")
        return model


class PickleAWSBucketPersistence(ModelPersistence):
    """AWS S3 bucket persistence using pickle serialization."""
    
    def __init__(self, bucket_name: str):
        """
        Initialize AWS S3 persistence with pickle serialization.
        
        Uses environment variables for configuration:
        - AWS_DEFAULT_REGION or AWS_REGION: AWS region (defaults to 'us-east-1')
        - AWS_ACCESS_KEY_ID: AWS access key ID (optional, can use IAM roles)
        - AWS_SECRET_ACCESS_KEY: AWS secret access key (optional, can use IAM roles)
        - AWS_PROFILE: AWS profile name (optional)
        
        Args:
            bucket_name: Name of the S3 bucket
        """
        self.bucket_name = bucket_name
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
    
    def save(self, model: Any, path: str) -> None:
        """Save model to AWS S3 using pickle."""
        client = self._get_client()
        
        # Serialize model to bytes using pickle
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)
        
        # Upload to S3
        client.upload_fileobj(buffer, self.bucket_name, path)
        print(f"Model saved to AWS S3 (pickle): s3://{self.bucket_name}/{path}")
    
    def load(self, path: str) -> Any:
        """Load model from AWS S3 using pickle."""
        client = self._get_client()
        
        # Download from S3
        buffer = io.BytesIO()
        client.download_fileobj(self.bucket_name, path, buffer)
        buffer.seek(0)
        
        # Deserialize model using pickle
        model = pickle.load(buffer)
        print(f"Model loaded from AWS S3 (pickle): s3://{self.bucket_name}/{path}")
        return model


class PickleLocalFilePersistence(ModelPersistence):
    """Local file system persistence using pickle serialization."""
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize local file persistence with pickle serialization.
        
        Args:
            base_path: Base directory for storing models
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
    
    def save(self, model: Any, path: str) -> None:
        """Save model to local file system using pickle."""
        full_path = os.path.join(self.base_path, path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Save model using pickle
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved locally (pickle): {full_path}")
    
    def load(self, path: str) -> Any:
        """Load model from local file system using pickle."""
        full_path = os.path.join(self.base_path, path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model not found at: {full_path}")
        
        with open(full_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded locally (pickle): {full_path}")
        return model


class TorchGCPBucketPersistence(ModelPersistence):
    """GCP bucket persistence using PyTorch serialization."""
    
    def __init__(self, bucket_name: str):
        """
        Initialize GCP bucket persistence with PyTorch serialization.
        
        Uses environment variables for authentication:
        - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file (optional)
        - Or uses default credentials (gcloud auth, service account, etc.)
        
        Args:
            bucket_name: Name of the GCP bucket
        """
        self.bucket_name = bucket_name
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
    
    def save(self, model: Any, path: str) -> None:
        """Save model to GCP bucket using PyTorch serialization."""
        client, bucket = self._get_client()
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for torch persistence")
        
        # Serialize model to bytes using torch
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        
        # Upload to GCP bucket
        blob = bucket.blob(path)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        print(f"Model saved to GCP bucket (torch): gs://{self.bucket_name}/{path}")
    
    def load(self, path: str) -> Any:
        """Load model from GCP bucket using PyTorch serialization."""
        client, bucket = self._get_client()
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for torch persistence")
        
        # Download from GCP bucket
        blob = bucket.blob(path)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        
        # Deserialize model using torch
        model = torch.load(buffer, map_location='cpu')
        print(f"Model loaded from GCP bucket (torch): gs://{self.bucket_name}/{path}")
        return model


class TorchAWSBucketPersistence(ModelPersistence):
    """AWS S3 bucket persistence using PyTorch serialization."""
    
    def __init__(self, bucket_name: str):
        """
        Initialize AWS S3 persistence with PyTorch serialization.
        
        Uses environment variables for configuration:
        - AWS_DEFAULT_REGION or AWS_REGION: AWS region (defaults to 'us-east-1')
        - AWS_ACCESS_KEY_ID: AWS access key ID (optional, can use IAM roles)
        - AWS_SECRET_ACCESS_KEY: AWS secret access key (optional, can use IAM roles)
        - AWS_PROFILE: AWS profile name (optional)
        
        Args:
            bucket_name: Name of the S3 bucket
        """
        self.bucket_name = bucket_name
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
    
    def save(self, model: Any, path: str) -> None:
        """Save model to AWS S3 using PyTorch serialization."""
        client = self._get_client()
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for torch persistence")
        
        # Serialize model to bytes using torch
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        
        # Upload to S3
        client.upload_fileobj(buffer, self.bucket_name, path)
        print(f"Model saved to AWS S3 (torch): s3://{self.bucket_name}/{path}")
    
    def load(self, path: str) -> Any:
        """Load model from AWS S3 using PyTorch serialization."""
        client = self._get_client()
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for torch persistence")
        
        # Download from S3
        buffer = io.BytesIO()
        client.download_fileobj(self.bucket_name, path, buffer)
        buffer.seek(0)
        
        # Deserialize model using torch
        model = torch.load(buffer, map_location='cpu')
        print(f"Model loaded from AWS S3 (torch): s3://{self.bucket_name}/{path}")
        return model


class TorchLocalFilePersistence(ModelPersistence):
    """Local file system persistence using PyTorch serialization."""
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize local file persistence with PyTorch serialization.
        
        Args:
            base_path: Base directory for storing models
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
    
    def save(self, model: Any, path: str) -> None:
        """Save model to local file system using PyTorch serialization."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for torch persistence")
        
        full_path = os.path.join(self.base_path, path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Save model using torch
        torch.save(model, full_path)
        
        print(f"Model saved locally (torch): {full_path}")
    
    def load(self, path: str) -> Any:
        """Load model from local file system using PyTorch serialization."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for torch persistence")
        
        full_path = os.path.join(self.base_path, path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model not found at: {full_path}")
        
        model = torch.load(full_path, map_location='cpu')
        
        print(f"Model loaded locally (torch): {full_path}")
        return model


# Backward compatibility aliases (deprecated)
GCPBucketPersistence = PickleGCPBucketPersistence
AWSBucketPersistence = PickleAWSBucketPersistence
LocalFilePersistence = PickleLocalFilePersistence 