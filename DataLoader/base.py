"""Base classes and enums for data loaders."""

import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple


class DataSourceType(Enum):
    """Enum for different data source types."""
    CSV_FILE = "csv_file"
    GCP_CSV_FILE = "gcp_csv_file"
    NEWSGROUPS = "newsgroups"


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, list]:
        """
        Load data from the specified source.
        
        Returns:
            Tuple of (DataFrame, target_names) where:
            - DataFrame has columns ['text', 'target']
            - target_names is a list of class names
        """
        pass
    
    @abstractmethod
    def get_data_info(self) -> dict:
        """
        Get information about the loaded data.
        
        Returns:
            dict: Information about the data source
        """
        pass 