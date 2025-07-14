"""Experiments Package for Text Classification.

This package provides experiment orchestration functionality
for running multiple model comparisons and generating results.
"""

from .experiments import run_experiments

__all__ = [
    'run_experiments'
] 