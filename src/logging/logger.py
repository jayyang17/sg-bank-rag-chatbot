import os
import sys
import logging
from datetime import datetime

# Generate timestamp for folder and log file
current_date = datetime.now().strftime("%Y-%m-%d")  # Folder name (YYYY-MM-DD)
current_time = datetime.now().strftime("%H%M%S")  # Time for log filename (HHMMSS)

# Define log directory structure
log_dir = os.path.join("logs", current_date)  # logs/YYYY-MM-DD/
os.makedirs(log_dir, exist_ok=True)  # Create directory if not exists

# Define log file path
log_filename = f"log_{current_date}_{current_time}.log"  # log_YYYY-MM-DD_HHMMSS.log
log_filepath = os.path.join(log_dir, log_filename)

# Logging format
logging_str = "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    datefmt="%Y-%m-%d %H:%M:%S",  # Timestamp format in logs
    handlers=[
        logging.FileHandler(log_filepath),  # Save logs to file
        logging.StreamHandler(sys.stdout)  # Show logs in terminal
    ]
)

# Create logger instance
logger = logging.getLogger("email_license_logger")
