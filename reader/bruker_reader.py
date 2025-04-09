import pathlib
import warnings
import re

import typing as tp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BaselineRemoval import BaselineRemoval
from scipy.optimize import curve_fit
from scipy.signal import lfilter

import struct
import re


def _parse_key_value(line: str) -> tuple[str, str]:
    """Extracts key and value from a line, handling multiple spaces or tabs."""
    parts = re.split(r'\s{2,}|\t', line, maxsplit=1)
    key = parts[0].strip()
    value = parts[1].strip() if len(parts) > 1 else ''
    return key, value


def _parse_comma_separated_values(value):
    """Parses comma-separated values into a list of numbers."""
    values = value.replace('\n', '').split(',')
    parsed_values = []

    for v in values:
        v = v.strip()
        if v.replace('.', '', 1).isdigit():  # Check if it's a number
            parsed_values.append(float(v) if '.' in v else int(v))
        else:
            parsed_values.append(v)  # Keep as a string if it's not a number

    return parsed_values


def _parse_numeric_value(value):
    """Extracts numbers and units from a single value."""
    match = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z/%]*)", value)
    if match:
        num, unit = match.groups()
        value = float(num) if '.' in num or 'e' in num.lower() else int(num)
        return {'value': value, 'unit': unit} if unit else value
    return value


def _handle_dvc_lines(key, value):
    """Handles special `.DVC` metadata lines by restructuring the key."""
    key_parts = key.split()
    if len(key_parts) > 1:
        key = f"{key_parts[0]}_{key_parts[1]}"
        value = key_parts[2] if len(key_parts) > 2 else ''
    return key, value


def read_dsc(filename):
    """Reads and cleans a Bruker DSC file, returning a structured dictionary."""
    metadata = {}
    current_key = None  # Track current pulse key

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Ignore empty lines and headers
            if not line or line.startswith('*') or line.startswith('#'):
                continue

            # Check if it's a continuation of pulse data (PsdX)
            if current_key and re.match(r'^\d+(,\d+)*$', line):
                metadata[current_key].extend(_parse_comma_separated_values(line))
                continue

            key, value = _parse_key_value(line)

            # Handle `.DVC` special cases
            if key.startswith('.DVC'):
                key, value = _handle_dvc_lines(key, value)

            # Handle pulse data (PsdX)
            if key.startswith('Psd') and ',' in value:
                metadata[key] = _parse_comma_separated_values(value)
                current_key = key  # Store key for multi-line handling
            else:
                metadata[key] = _parse_numeric_value(value)
                current_key = None  # Reset pulse tracking

    return metadata


def read_dta(filepath, metadata):
    """
    Read Bruker .DTA file with complex data for EPR spectroscopy
    Parameters:
    - x_values: numpy array of x-axis values
    """
    # Determine data format based on metadata

    # Determine endianness ('>' for big-endian, '<' for little-endian)
    endian = '>' if metadata['BSEQ'] == 'BIG' else '<'

    # 'D' means double precision (64-bit float), so define the data type accordingly.
    if metadata["IRFMT"] == "D":
        dtype = np.dtype(endian + 'f8')
    else:
        dtype = np.dtype(endian + 'f4')
    raw_data = np.fromfile(filepath, dtype=dtype)

    # If the data is complex (as indicated by IKKF), it is stored as interleaved real and imaginary parts.
    if metadata['IKKF'] == 'CPLX':
        # Expect two numbers per data point (real and imaginary).
        raw_data = raw_data.reshape(-1, 2)
        data = raw_data[:, 0] + 1j * raw_data[:, 1]
    else:
        data = raw_data
    x_axis = np.linspace(metadata['XMIN'], metadata['XMIN'] + metadata['XWID'], metadata['XPTS'])
    return {
        'x_values': x_axis,
        'y_values': data,
    }


def read_bruker_data(path: str | pathlib.Path) -> tuple[dict[str, tp.Any], dict[str, np.array]]:
    """
    :param path: path to bruker file
    :return: metadata and the results
    """
    path = pathlib.Path(path)
    path_dta = path.with_suffix(".dta")
    path_dsc = path.with_suffix(".dsc")
    metadata = read_dsc(path_dsc)
    data = read_dta(path_dta, metadata=metadata)
    return metadata, data
