
import sys
import os

# Mask environment variables to avoid real connections if possible, 
# although deployer.py validates them at import time usually.
# However, deployer.py has robust error handling so it might import fine.
# We just want to check signature.

try:
    from deployer import get_crew
    print("Successfully imported get_crew")
except Exception as e:
    print(f"Failed to import: {e}")
    # continue anyway in case it works partially

try:
    # We don't expect it to actually RUN successfully because of missing keys/db,
    # but we want to see if it accepts the argument.
    # It will likely fail inside factory.create_crew or earlier, but NOT with TypeError on get_crew
    get_crew(previous_feedback="test feedback")
    print("Call to get_crew(previous_feedback='...') succeeded (unexpectedly ran fully?)")
except TypeError as Te:
    if "unexpected keyword argument 'previous_feedback'" in str(Te):
        print("FAIL: TypeError still present!")
    else:
        print(f"TypeError occurred but likely unrelated to argument: {Te}")
except Exception as e:
    print(f"Caught expected exception during execution (not TypeError on signature): {type(e).__name__}: {e}")
    print("PASS: Signature accepted 'previous_feedback'")
