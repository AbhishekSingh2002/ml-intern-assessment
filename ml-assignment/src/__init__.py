"""
Trigram Language Model (TLM)

A Python package for training and using trigram language models for text generation.

This package provides:
- TrigramModel: A class for training and using trigram language models
- Command-line interface for training and text generation
- Utility functions for text processing

Example usage:
    >>> from trigram_lm import TrigramModel
    >>> model = TrigramModel()
    >>> model.fit("This is some example text.")
    >>> generated_text = model.generate()
"""

from .ngram_model import TrigramModel

__version__ = '0.1.0'
__author__ = 'Your Name <your.email@example.com>'
__license__ = 'MIT'

__all__ = [
    'TrigramModel',
    '__version__',
    '__author__',
    '__license__',
]
