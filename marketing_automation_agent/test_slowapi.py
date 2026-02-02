import slowapi
from slowapi.extension import Limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from fastapi import FastAPI

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
def read_root(): return {"Hello": "World"}

if __name__ == "__main__":
    print("Starting test server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
