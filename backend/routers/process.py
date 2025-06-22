from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import numpy as np
import os
import requests
import io

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
import google.auth.transport.requests
from google.oauth2 import service_account

from schemas.request import RequestMesh
from utils.gcs import GCSHandler
from utils.postprocessing import taubin_smoothing, laplacian_smoothing

load_dotenv()

MESH_URL = os.getenv("MESH_ENDPOINT", "")
auth_req = google.auth.transport.requests.Request()

router = APIRouter(
    prefix="/process",
    tags=["process"],
    responses={404: {"description": "Not found"}}
)

@router.post("/mesh")
async def infer_mesh(request: RequestMesh):
    """
    Process a mesh file and return the processed mesh.
    """
    try:
        print("Received request data:", request.dict())
        if not request.file_path:
            raise HTTPException(status_code=400, detail="File path is required")
        
        creds_token_id = service_account.IDTokenCredentials.from_service_account_file(
            "./secrets/key.json",
            target_audience=MESH_URL
        )
        creds_token_id.refresh(auth_req)
        TOKEN_ID = creds_token_id.token
        
        file_path = request.file_path
        payload = {
            "file_path": file_path,
            "res": 128
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN_ID}"
        }
        response = requests.post(f"{MESH_URL}/predict", headers=headers, json=payload)
        
        data = response.json()
        mesh_file_path = data.get("file_path")
        
        if not mesh_file_path:
            raise HTTPException(status_code=500, detail="Mesh processing failed or no mesh file URL returned")
        
        gcs = GCSHandler()
        
        file_extension = os.path.splitext(mesh_file_path)[1]
        tmp_file = NamedTemporaryFile(delete=False, suffix=file_extension)
        local_path, blob = gcs.download_file(mesh_file_path, tmp_file.name)
        
        print(f"Smoothing mesh with algorithm: {request.smoothing_algorithm}")
        if request.smoothing_algorithm == "taubin":
            smoothed_path = NamedTemporaryFile(delete=False, suffix=file_extension).name
            taubin_smoothing(local_path, smoothed_path)
            local_path = smoothed_path
        if request.smoothing_algorithm == "laplacian":
            smoothed_path = NamedTemporaryFile(delete=False, suffix=file_extension).name
            laplacian_smoothing(local_path, smoothed_path, iterations=request.smoothing_iterations)
            local_path = smoothed_path

        return FileResponse(
            path=local_path,
            media_type="application/octet-stream",
            filename=os.path.basename(local_path)
        )
    except Exception as e:
        print(f"Error during upsampling: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
    # test local
    # file_path = "./data/test_8192_mesh.ply"
    
@router.get("/proxy-download")
def proxy_file(url: str):
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch file from GCS")

    return StreamingResponse(io.BytesIO(r.content), media_type="application/octet-stream")
    