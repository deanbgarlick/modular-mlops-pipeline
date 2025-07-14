"""DataLoader Package for Text Classification.

This package provides a modular approach to data loading for text classification,
supporting multiple data sources through a clean interface.
"""

from .base import DataLoader, DataSourceType
from .csv_loader import CSVDataLoader
from .newsgroups_loader import NewsgroupsDataLoader
from .factory import create_data_loader

__all__ = [
    'DataLoader',
    'DataSourceType',
    'CSVDataLoader',
    'NewsgroupsDataLoader',
    'create_data_loader'
] 