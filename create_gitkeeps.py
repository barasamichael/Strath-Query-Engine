import os
from pathlib import Path

# Directories to create .gitkeep files in
dirs_for_gitkeep = [
    "data/raw",
    "data/processed",
    "data/chunks",
    "data/embeddings",
    "data/metadata",
    "database/vector_store",
    "database/relational",
    "models/embeddings",
    "models/llm",
    "models/intent",
]

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Create .gitkeep files
for dir_path in dirs_for_gitkeep:
    full_path = ROOT_DIR / dir_path

    # Ensure directory exists
    if not full_path.exists():
        os.makedirs(full_path)

    # Create .gitkeep file
    gitkeep_file = full_path / ".gitkeep"
    if not gitkeep_file.exists():
        with open(gitkeep_file, "w") as f:
            f.write("# This file ensures the directory is tracked by Git")
        print(f"Created .gitkeep in {dir_path}")

print("Done creating .gitkeep files")
