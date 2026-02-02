import traceback
import sys

print("Testing imports...")
try:
    print("Importing deployer...")
    import deployer
    print("Importing slowapi.Limiter...")
    from slowapi import Limiter
    print("Success")
except Exception:
    traceback.print_exc()
except ImportError:
    traceback.print_exc()
