"""
SoliReward Data Module

This module provides data loading and processing utilities:
- Dataset creation from JSON files
- Data format conversion (pair format, ms-swift format to OpenAI format)
"""

from .reward_data import (
    load_json,
    load_pair_json_data,
    convert_pair_json_data_to_openai_format,
    convert_ms_swift_data_to_openai_format,
    create_dataset_from_json,
)

__all__ = [
    "load_json",
    "load_pair_json_data",
    "convert_pair_json_data_to_openai_format",
    "convert_ms_swift_data_to_openai_format",
    "create_dataset_from_json",
]
