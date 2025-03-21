import os 
import yaml
from src.logging.logger import logging
from ensure import ensure_annotations
from pathlib import Path
from datetime import datetime
import pandas as pd

@ensure_annotations
def read_yaml(path_to_yaml: Path):
    """reads yaml file and return

    Args:
        path_to_yaml (str): path input
    
    Raises:
        ValueError: if yaml file is empty
        e: empty file
    
    Returns:
        ConfigBox: ConfigBox type

    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded succesfully")
            return content
        
    except ValueError:
        raise ValueError("yaml file is empty")
    
    except Exception as e:
        raise e

@ensure_annotations
def write_yaml(data, path_to_yaml: Path):
    """
    Writes data to a YAML file.
    """
    try:
        with open(path_to_yaml, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
            logging.info(f"yaml file updated")
    except ValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


def save_to_csv(data:dict, output_path, filename):
    """Save dictionary to a CSV file using Pandas."""

    os.makedirs(output_path, exist_ok=True)  # Ensure directory exists
    if not output_path:
        logging.error("Output path or shared path is missing!")
        return
    if data:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        timestamp = datetime.now().strftime("%Y%m%d")

        output_file = os.path.join(output_path, f"{filename}_{timestamp}.csv")
        df.to_csv(output_file, index=False)

        logging.info(f"Audit logs saved to {output_file}")
    else:
        logging.error("No data to save")