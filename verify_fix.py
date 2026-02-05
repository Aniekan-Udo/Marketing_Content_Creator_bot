
import os
import sys

# Add current directory to path so we can import deployer
sys.path.append(os.getcwd())

# Mock environment variables needed for imports
os.environ["POSTGRES_URI"] = "postgres://user:pass@localhost:5432/db"
os.environ["POSTGRES_ASYNC_URI"] = "postgresql+asyncpg://user:pass@localhost:5432/db"
os.environ["GROQ_API_KEY"] = "mock_key"
os.environ["TAVILY_API_KEY"] = "mock_key"

def verify_fix():
    print("Verifying fix for hardcoded content_type...")
    
    with open("deployer.py", "r") as f:
        content = f.read()
    
    # Check 1: Check inputs in crew.kickoff
    if '"content_type": content_type' in content and 'result = crew.kickoff(inputs={' in content:
        print("PASS: 'content_type' variable added to crew inputs.")
    else:
        print("FAIL: 'content_type' variable NOT found in crew inputs.")

    # Check 2: Check creative_strategy_task prompt
    if 'content_type": "{content_type}"' in content:
        print("PASS: Dynamic '{content_type}' found in task prompts.")
    else:
        print("FAIL: Dynamic '{content_type}' NOT found in task prompts (or fewer occurrences than expected).")
        
    # Check 3: Ensure "blog" hardcodes are gone from key areas
    # We look for the specific lines we changed
    if 'content_type": "blog"' in content:
        # We need to be careful, "blog" might still exist as a fallback or in other valid places
        # Let's check the specific lines we aimed to change by looking for adjacent context
        if 'Use learning_memory tool:\n                    - {{"action": "get_rejected", "content_type": "blog"}}' in content:
             print("FAIL: Hardcoded 'blog' still found in creative_strategy_task.")
        elif '1. {{"content_type": "blog", "query": "tone characteristics"}}' in content:
             print("FAIL: Hardcoded 'blog' still found in brand_analysis_task.")
        else:
             print("PASS: Hardcoded 'blog' removed from target task prompts.")
    else:
        print("PASS: Hardcoded 'blog' removed from target task prompts.")

if __name__ == "__main__":
    verify_fix()
