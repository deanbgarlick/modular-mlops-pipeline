"""GCP CSV data loader implementation."""

import pandas as pd
import io
from typing import Tuple, Optional

from .base import DataLoader


class GCPCSVDataLoader(DataLoader):
    """Data loader for CSV files stored in Google Cloud Storage buckets."""
    
    def __init__(self, bucket_name: str, blob_path: str, 
                 text_column: str = "customer_review", target_column: str = "return", 
                 sep: str = "\t", credentials_path: Optional[str] = None):
        """
        Initialize GCP CSV data loader.
        
        Args:
            bucket_name: Name of the GCP bucket
            blob_path: Path to the CSV file in the bucket (e.g., "data/dataset.csv")
            text_column: Name of the column containing text data
            target_column: Name of the column containing target labels
            sep: Delimiter for the CSV file
            credentials_path: Path to GCP credentials JSON file (optional)
        """
        self.bucket_name = bucket_name
        self.blob_path = blob_path
        self.text_column = text_column
        self.target_column = target_column
        self.sep = sep
        self.credentials_path = credentials_path
        self.data = None
        self.target_names = None
        self._client = None
        self._bucket = None
    
    def _get_client(self):
        """Lazy initialization of GCP client."""
        if self._client is None:
            try:
                from google.cloud import storage
                if self.credentials_path:
                    self._client = storage.Client.from_service_account_json(self.credentials_path)
                else:
                    self._client = storage.Client()
                self._bucket = self._client.bucket(self.bucket_name)
            except ImportError:
                raise ImportError("google-cloud-storage package is required for GCP CSV loader. "
                                "Install with: pip install google-cloud-storage")
        return self._client
    
    def load_data(self) -> Tuple[pd.DataFrame, list]:
        """Load data from CSV file in GCP bucket."""
        print(f"Loading data from GCP bucket: gs://{self.bucket_name}/{self.blob_path}")
        
        # Get GCP client and bucket
        self._get_client()
        
        # Download CSV file from GCP bucket
        blob = self._bucket.blob(self.blob_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"CSV file not found: gs://{self.bucket_name}/{self.blob_path}")
        
        # Download the file content
        csv_content = blob.download_as_text()
        
        # Load CSV from string content
        raw_data = pd.read_csv(io.StringIO(csv_content), sep=self.sep, index_col=0)
        
        # Clean data - remove rows with missing values
        raw_data = raw_data.dropna(subset=[self.text_column, self.target_column])
        
        # Create standardized DataFrame
        self.data = pd.DataFrame({
            'text': raw_data[self.text_column],
            'target': raw_data[self.target_column].astype(int)
        })
        
        # Create target names
        unique_targets = sorted(self.data['target'].unique())
        self.target_names = [f"class_{target}" for target in unique_targets]
        
        print(f"GCP CSV data loaded: {self.data.shape[0]} samples, {len(self.target_names)} classes")
        print(f"Target distribution:")
        print(self.data['target'].value_counts().sort_index())
        
        return self.data, self.target_names
    
    def get_data_info(self) -> dict:
        """Get information about the GCP CSV data."""
        if self.data is None:
            return {"error": "Data not loaded yet"}
        
        return {
            "data_source": "gcp_csv_file",
            "bucket_name": self.bucket_name,
            "blob_path": self.blob_path,
            "full_path": f"gs://{self.bucket_name}/{self.blob_path}",
            "text_column": self.text_column,
            "target_column": self.target_column,
            "separator": self.sep,
            "credentials_path": self.credentials_path,
            "n_samples": len(self.data),
            "n_classes": len(self.target_names) if self.target_names else 0,
            "target_names": self.target_names,
            "class_distribution": self.data['target'].value_counts().sort_index().to_dict()
        } 