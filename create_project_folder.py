import os

# Base project folder
base_folder = "marketing_automation_agent"

# Folder structure
folders = [
    "config",
    "agents",
    "rag",
    "llm",
    "embeddings",
    "utils"
]

# Files to create in each folder
files = {
    "config": ["env.example", "settings.py"],
    "agents": ["researcher.py", "content_generator.py", "reviewer.py", "deployer.py"],
    "rag": ["vector_store.py", "data_loader.py"],
    "llm": ["groq_client.py"],
    "embeddings": ["hf_embeddings.py"],
    "utils": ["logger.py", "helpers.py"],
    "": ["main.py", "requirements.txt"]
}

# Create folders
for folder in folders:
    folder_path = os.path.join(base_folder, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")

# Create files
for folder, file_list in files.items():
    for file_name in file_list:
        file_path = os.path.join(base_folder, folder, file_name) if folder else os.path.join(base_folder, file_name)
        with open(file_path, "w") as f:
            f.write(f"# {file_name} - placeholder\n")
        print(f"Created file: {file_path}")

print("\n✅ Project structure created successfully!")