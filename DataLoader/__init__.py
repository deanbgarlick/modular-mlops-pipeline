"""DataLoader Package for Text Classification.

This package provides a modular approach to data loading for text classification,
supporting multiple data sources through a clean interface, including local CSV files,
GCP bucket CSV files, and newsgroups data.
"""

from .base import DataLoader, DataSourceType
from .local_csv_loader import LocalCSVDataLoader
from .gcp_csv_loader import GCPCSVDataLoader
from .newsgroups_loader import NewsgroupsDataLoader
from .factory import create_data_loader

__all__ = [
    'DataLoader',
    'DataSourceType',
    'LocalCSVDataLoader',
    'GCPCSVDataLoader',
    'NewsgroupsDataLoader',
    'create_data_loader'
] 