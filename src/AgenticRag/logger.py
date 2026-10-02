import os
import logging
from datetime import datetime

Log_Dir="logs"
os.makedirs(Log_Dir,exist_ok=True)
log_file = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
log_file_path=os.path.join(Log_Dir,log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO
)

logger=logging.getLogger(__file__)