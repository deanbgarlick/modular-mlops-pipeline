"""Newsgroups data loader implementation."""

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from typing import Tuple, List, Optional

from .base import DataLoader


class NewsgroupsDataLoader(DataLoader):
    """Data loader for 20 newsgroups dataset."""
    
    def __init__(self, categories: Optional[List[str]] = None, subset: str = 'train', 
                 shuffle: bool = True, random_state: int = 42):
        """
        Initialize newsgroups data loader.
        
        Args:
            categories: List of category names to load (None for all)
            subset: Dataset subset to load ('train', 'test', or 'all')
            shuffle: Whether to shuffle the data
            random_state: Random seed for reproducibility
        """
        self.categories = categories or ['alt.atheism', 'soc.religion.christian']
        self.subset = subset
        self.shuffle = shuffle
        self.random_state = random_state
        self.data = None
        self.target_names = None
    
    def load_data(self) -> Tuple[pd.DataFrame, list]:
        """Load data from 20 newsgroups dataset."""
        print(f"Loading binary text classification dataset from 20 newsgroups...")
        print(f"Categories: {self.categories}")
        
        # Load newsgroups data
        newsgroups_data = fetch_20newsgroups(
            subset=self.subset,
            categories=self.categories,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        # Create standardized DataFrame
        self.data = pd.DataFrame({
            'text': newsgroups_data.data,
            'target': newsgroups_data.target
        })
        
        # Store target names
        self.target_names = newsgroups_data.target_names
        
        print(f"Newsgroups data loaded: {self.data.shape[0]} samples, {len(self.target_names)} classes")
        print(f"Target classes: {self.target_names}")
        print(f"Target distribution:")
        print(self.data['target'].value_counts().sort_index())
        
        return self.data, self.target_names
    
    def get_data_info(self) -> dict:
        """Get information about the newsgroups data."""
        if self.data is None:
            return {"error": "Data not loaded yet"}
        
        return {
            "data_source": "newsgroups",
            "categories": self.categories,
            "subset": self.subset,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "n_samples": len(self.data),
            "n_classes": len(self.target_names) if self.target_names else 0,
            "target_names": self.target_names,
            "class_distribution": self.data['target'].value_counts().sort_index().to_dict()
        } 