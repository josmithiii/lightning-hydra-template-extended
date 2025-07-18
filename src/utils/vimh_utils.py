"""Utility functions for VIMH dataset metadata handling."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_vimh_metadata(data_dir: str) -> Dict:
    """Load VIMH dataset metadata from JSON file.

    :param data_dir: Path to dataset directory
    :return: Dictionary containing dataset metadata
    :raises FileNotFoundError: If metadata file doesn't exist
    :raises json.JSONDecodeError: If metadata file is malformed
    """
    metadata_file = Path(data_dir) / 'vimh_dataset_info.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"VIMH metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata


def get_parameter_names_from_metadata(data_dir: str) -> List[str]:
    """Get parameter names from VIMH dataset metadata.

    :param data_dir: Path to dataset directory
    :return: List of parameter names
    """
    metadata = load_vimh_metadata(data_dir)
    return metadata.get('parameter_names', [])


def get_parameter_ranges_from_metadata(data_dir: str) -> Dict[str, Tuple[float, float]]:
    """Get parameter ranges from VIMH dataset metadata.

    :param data_dir: Path to dataset directory
    :return: Dictionary mapping parameter names to (min, max) tuples
    """
    metadata = load_vimh_metadata(data_dir)
    parameter_ranges = {}

    if 'parameter_names' in metadata and 'parameter_mappings' in metadata:
        param_names = metadata['parameter_names']
        param_mappings = metadata['parameter_mappings']

        for param_name in param_names:
            if param_name in param_mappings:
                mapping = param_mappings[param_name]
                parameter_ranges[param_name] = (mapping['min'], mapping['max'])

    return parameter_ranges


def get_heads_config_from_metadata(data_dir: str) -> Dict[str, int]:
    """Get heads configuration from VIMH dataset metadata.

    :param data_dir: Path to dataset directory
    :return: Dictionary mapping head names to number of classes
    """
    metadata = load_vimh_metadata(data_dir)
    heads_config = {}

    if 'parameter_names' in metadata and 'parameter_mappings' in metadata:
        param_names = metadata['parameter_names']
        param_mappings = metadata['parameter_mappings']

        for param_name in param_names:
            if param_name in param_mappings:
                # For continuous parameters, use 256 classes (0-255 quantization)
                heads_config[param_name] = 256

    return heads_config
