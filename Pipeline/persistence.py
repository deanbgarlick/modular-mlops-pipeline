"""Persistence interfaces and implementations for pipelines."""

import json
import os
import io
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import storage


class PipelinePersistence(ABC):
    """Abstract base class for pipeline persistence."""
    
    @abstractmethod
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> None:
        """Save pipeline metadata."""
        pass
    
    @abstractmethod
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load pipeline metadata."""
        pass


class LocalPipelinePersistence(PipelinePersistence):
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


class GCPPipelinePersistence(PipelinePersistence):
    """GCP bucket persistence for Pipeline metadata using JSON."""
    
    def __init__(self, bucket_name: str):
        """
        Initialize GCP bucket persistence for pipeline metadata.
        
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
    
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> None:
        """Save pipeline metadata to GCP bucket as JSON."""
        client, bucket = self._get_client()
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata, indent=2)
        
        # Upload to GCP bucket
        blob_path = f"{path}.json"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(metadata_json, content_type='application/json')
        
        print(f"Pipeline metadata saved to GCP bucket: gs://{self.bucket_name}/{blob_path}")
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load pipeline metadata from GCP bucket."""
        client, bucket = self._get_client()
        
        # Download from GCP bucket
        blob_path = f"{path}.json"
        blob = bucket.blob(blob_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"Pipeline metadata not found at: gs://{self.bucket_name}/{blob_path}")
        
        metadata_json = blob.download_as_text()
        metadata = json.loads(metadata_json)
        
        print(f"Pipeline metadata loaded from GCP bucket: gs://{self.bucket_name}/{blob_path}")
        return metadata 