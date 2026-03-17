import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")

project_name = "src"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/settings.py",
    f"{project_name}/models/__init__.py",
    f"{project_name}/models/embeddings.py",
    f"{project_name}/models/llm_service.py",
    f"{project_name}/models/vector_store.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/rag_chain.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/image_utils.py",
    f"{project_name}/utils/response_formatter.py",
    "data/"
    "app.py",
    "ingest.py",
    ".env",
    "requirements.txt",
]

for file_path in list_of_files:
    file_path =  Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")

