
import os
import sys

# Add current directory to path so we can import deployer
sys.path.append(os.getcwd())

# Mock environment variables needed for imports
os.environ["POSTGRES_URI"] = "postgres://user:pass@localhost:5432/db"
os.environ["POSTGRES_ASYNC_URI"] = "postgresql+asyncpg://user:pass@localhost:5432/db"
os.environ["GROQ_API_KEY"] = "mock_key"
os.environ["TAVILY_API_KEY"] = "mock_key"

try:
    from deployer import ContentCrewFactory
except ImportError:
    # If imports fail due to missing dependencies, we might need to mock them or just read the file differently.
    # But since we are in the environment, it should work.
    pass

def check_task_description():
    # We can't easily instantiate ContentCrewFactory without real objects because of the type hints and inner logic
    # But we can inspect the class or just read the file. 
    # Actually, instantiating might fail if it tries to connect to DB.
    # Let's simple read the file content again and print the specific lines to be 100% sure of what we saw.
    
    with open("deployer.py", "r") as f:
        content = f.read()
    
    start_marker = 'creative_strategy_task = Task('
    end_marker = 'agent=creative_strategist,'
    
    start_idx = content.find(start_marker)
    if start_idx != -1:
        end_idx = content.find(end_marker, start_idx)
        task_def = content[start_idx:end_idx]
        print("Found Task Definition:\n")
        print(task_def)
        
        if 'content_type": "blog"' in task_def:
            print("\nCONFIRMED: Hardcoded 'blog' content type found.")
        else:
            print("\nNOT FOUND: Hardcoded 'blog' content type not found.")
            
if __name__ == "__main__":
    check_task_description()
