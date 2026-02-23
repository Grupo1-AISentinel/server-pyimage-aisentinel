from fastapi import FastAPI
from api.routes import router as api_router
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="AI Sentinel API")

app.include_router(api_router, prefix="/pyimage/api/v1")

PORT = int(os.getenv("PORT", 8000))

@app.get("/")
def home():
    return {"project": "AI Sentinel", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)