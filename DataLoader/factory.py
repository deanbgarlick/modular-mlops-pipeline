"""Factory function for creating data loaders."""

from .base import DataLoader, DataSourceType
from .local_csv_loader import LocalCSVDataLoader
from .gcp_csv_loader import GCPCSVDataLoader
from .newsgroups_loader import NewsgroupsDataLoader


def create_data_loader(data_source_type: DataSourceType, **kwargs) -> DataLoader:
    """Create and return the appropriate data loader.
    
    Args:
        data_source_type: The type of data source to load
        **kwargs: Additional arguments to pass to the loader constructor
        
    Returns:
        DataLoader: The created data loader instance
        
    Raises:
        ValueError: If data_source_type is not supported
    """
    if data_source_type == DataSourceType.CSV_FILE:
        return LocalCSVDataLoader(**kwargs)
    elif data_source_type == DataSourceType.GCP_CSV_FILE:
        return GCPCSVDataLoader(**kwargs)
    elif data_source_type == DataSourceType.NEWSGROUPS:
        return NewsgroupsDataLoader(**kwargs)
    else:
        raise ValueError(f"Unknown data source type: {data_source_type}") 