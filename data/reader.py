"""Raw data file reader for spray penetration experiments.

Reads .dat/.txt files containing time-penetration curves and physical parameters.
Applies the same normalisation as the original data_read.py pipeline.
"""

import re
from pathlib import Path

import chardet

def detect_encoding(file_path: str) -> str:
    """Detect file encoding by reading the first 1024 bytes."""
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(1024))
    return result["encoding"]


def process_single_file(file_path: str) -> dict:
    """Parse a single .dat/.txt file into time, penetration and normalised physical params.

    Returns
    -------
    dict with keys: time, penetration, params, missing_params, kbc, file_path
    """
    time_list: list[float] = []
    penetration_list: list[float] = []
    params_list: list[float] = []
    kbc_params: dict[str, float] = {}

    # --- detect & read ---
    try:
        encoding = detect_encoding(file_path)
    except Exception:
        encoding = "utf-8"

    try:
        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

    reading_time_data = False
    reading_params = False
    params_found: set[str] = set()
    found_prediction_input = False
    found_kbc_section = False

    # regex patterns for target parameters (with units)
    target_params: dict[str, str] = {
        r"ambient_density\s*\(g/cm3\)": "ambient_density",
        r"ambient_temperature\s*\(K\)": "ambient_temperature",
        r"fuel_temperature\s*\(K\)": "fuel_temperature",
        r"injection_pressure\s*\(MPa\)": "injection_pressure",
        r"nozzle_radius\s*\(cm\)": "nozzle_radius",
    }

    for line in lines:
        original_line = line
        line = line.split("!", 1)[0].strip()  # remove Fortran-style comments

        # --- state transitions ---
        if (
            re.search(r"Predicted\s+Penetration", original_line, re.IGNORECASE)
            or re.search(r"Experimental\s+Penetration", original_line, re.IGNORECASE)
            or re.search(r"Predict\s+Output", original_line, re.IGNORECASE)
            or re.search(r"Fit\s+Output", original_line, re.IGNORECASE)
        ):
            reading_time_data = True
            continue
        elif re.search(r"Prediction\s+Input", original_line, re.IGNORECASE) or re.search(
            r"Params", original_line, re.IGNORECASE
        ):
            reading_time_data = False
            reading_params = True
            found_prediction_input = True
            continue
        elif re.search(r"(k|b|c)\s*:", original_line, re.IGNORECASE):
            found_kbc_section = True
            k_match = re.search(
                r"k\s*:\s*([-+]?\d*\.\d+e[-+]\d+|[-+]?\d+\.\d+|[-+]?\d+)",
                original_line,
                re.IGNORECASE,
            )
            b_match = re.search(
                r"b\s*:\s*([-+]?\d*\.\d+e[-+]\d+|[-+]?\d+\.\d+|[-+]?\d+)",
                original_line,
                re.IGNORECASE,
            )
            c_match = re.search(
                r"c\s*:\s*([-+]?\d*\.\d+e[-+]\d+|[-+]?\d+\.\d+|[-+]?\d+)",
                original_line,
                re.IGNORECASE,
            )
            if k_match:
                kbc_params["k"] = float(k_match.group(1))
            if b_match:
                kbc_params["b"] = float(b_match.group(1))
            if c_match:
                kbc_params["c"] = float(c_match.group(1)) * 0.01

        # --- read (time, penetration) pairs ---
        if reading_time_data and line:
            parts = re.findall(r"[-+]?\d*\.\d+e[-+]\d+|[-+]?\d+\.\d+|[-+]?\d+", line.lower())
            if len(parts) >= 2:
                time_list.append(float(parts[0]))
                penetration_list.append(float(parts[1]))

        # --- read physical parameters (normalised on the fly) ---
        if reading_params and line and not found_kbc_section:
            for param_pattern, param_name in target_params.items():
                param_match = re.search(
                    param_pattern
                    + r"\s*:\s*([-+]?\d*\.\d+e[-+]\d+|[-+]?\d+\.\d+|[-+]?\d+)",
                    line,
                    re.IGNORECASE,
                )
                if param_match:
                    if param_name == "ambient_density":
                        param_value = (float(param_match.group(1)) - 1.93e-2) / 0.0134
                        params_list.append(param_value)
                        params_found.add(param_name)
                    elif param_name == "ambient_temperature":
                        param_value = (float(param_match.group(1)) - 600) / 577
                        params_list.append(param_value)
                        params_found.add(param_name)
                    elif param_name == "fuel_temperature":
                        param_value = float(param_match.group(1))
                        params_found.add(param_name)
                    elif param_name == "nozzle_radius":
                        param_value = (float(param_match.group(1)) - 0.009) / 0.0055
                        params_list.append(param_value)
                        params_found.add(param_name)
                    elif param_name == "injection_pressure":
                        param_value = (float(param_match.group(1)) - 100) / 84.25
                        params_list.append(param_value)
                        params_found.add(param_name)

    if not found_prediction_input:
        print(f"  Warning: no 'Prediction Input' marker found in {file_path}")

    missing_params = [p for p in target_params.values() if p not in params_found]

    return {
        "time": time_list,
        "penetration": penetration_list,
        "params": params_list,
        "missing_params": missing_params,
        "kbc": kbc_params,
        "file_path": file_path,
    }


def process_folder(folder_path: str) -> list[dict]:
    """Batch-read all .txt/.dat files in a folder."""
    folder = Path(folder_path)
    all_data: list[dict] = []
    for file_path in sorted(folder.glob("*")):
        if file_path.is_file() and file_path.suffix in (".txt", ".dat"):
            all_data.append(process_single_file(str(file_path)))
    return all_data


def process_file_list(folder_base: str, relative_paths: list[str]) -> list[dict]:
    """Read samples by a list of relative paths (e.g. from split config)."""
    folder = Path(folder_base)
    all_data: list[dict] = []
    for rel in relative_paths:
        rel = str(rel).replace("\\", "/")
        full = folder / rel
        if not full.exists():
            raise FileNotFoundError(f"Split file not found: {full}")
        all_data.append(process_single_file(str(full)))
    return all_data

