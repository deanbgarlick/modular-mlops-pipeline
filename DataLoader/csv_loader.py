"""CSV data loader implementation."""

import pandas as pd
from typing import Tuple

from .base import DataLoader


class CSVDataLoader(DataLoader):
    """Data loader for CSV files."""
    
    def __init__(self, file_path: str = "dataset.csv", text_column: str = "customer_review", 
                 target_column: str = "return", sep: str = "\t"):
        """
        Initialize CSV data loader.
        
        Args:
            file_path: Path to the CSV file
            text_column: Name of the column containing text data
            target_column: Name of the column containing target labels
            sep: Delimiter for the CSV file
        """
        self.file_path = file_path
        self.text_column = text_column
        self.target_column = target_column
        self.sep = sep
        self.data = None
        self.target_names = None
    
    def load_data(self) -> Tuple[pd.DataFrame, list]:
        """Load data from CSV file."""
        print(f"Loading data from CSV file: {self.file_path}")
        
        # Load CSV file
        raw_data = pd.read_csv(self.file_path, sep=self.sep, index_col=0)
        
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
        
        print(f"CSV data loaded: {self.data.shape[0]} samples, {len(self.target_names)} classes")
        print(f"Target distribution:")
        print(self.data['target'].value_counts().sort_index())
        
        return self.data, self.target_names
    
    def get_data_info(self) -> dict:
        """Get information about the CSV data."""
        if self.data is None:
            return {"error": "Data not loaded yet"}
        
        return {
            "data_source": "csv_file",
            "file_path": self.file_path,
            "text_column": self.text_column,
            "target_column": self.target_column,
            "separator": self.sep,
            "n_samples": len(self.data),
            "n_classes": len(self.target_names) if self.target_names else 0,
            "target_names": self.target_names,
            "class_distribution": self.data['target'].value_counts().sort_index().to_dict()
        } 