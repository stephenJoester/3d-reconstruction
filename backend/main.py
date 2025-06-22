from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import inference, process

app = FastAPI(
    title="3D Reconstruction API",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc",
    root_path="/api"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(inference.router)
app.include_router(process.router)

@app.get("/")
async def root(): 
    return {"message": "Hello World"}