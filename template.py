import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Project Name
project_name = "AgenticRag"

# List of files and folders
list_of_files = [

    # GitHub
    ".github/workflows/.gitkeep",

    # Source Code
    f"src/{project_name}/__init__.py",

    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/retriever.py",
    f"src/{project_name}/components/reranker.py",
    f"src/{project_name}/components/generator.py",

    # Agentic RAG
    f"src/{project_name}/agent/__init__.py",
    f"src/{project_name}/agent/state.py",
    f"src/{project_name}/agent/nodes.py",
    f"src/{project_name}/agent/router.py",
    f"src/{project_name}/agent/graph.py",

    # Pipelines
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/ingestion_pipeline.py",
    f"src/{project_name}/pipelines/rag_pipeline.py",

    # Utilities
    f"src/{project_name}/utils.py",
    f"src/{project_name}/Exception.py",
    f"src/{project_name}/logger.py",

    # Configuration
    "config/config.yaml",

    # Data
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    # Vector Database
    "vectorstore/.gitkeep",

    # Evaluation
    "evaluation/.gitkeep",

    # Notebook
    "notebooks/research.ipynb",

    # API
    "app/app.py",

    # Project Files
    "main.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "Dockerfile",
]

# Create folders and files
for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if not filepath.exists():
        with open(filepath, "w"):
            pass
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"{filepath} already exists")
