from fastapi import APIRouter, HTTPException, File, UploadFile

from google.oauth2 import service_account
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import google.auth.transport.requests

from utils.preprocessing import Preprocessor
from utils.gcs import GCSHandler
from utils.postprocessing import save_xyz_file, save_ply_file
from schemas.request import RequestUpsampling

import json
import os
import requests
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import numpy as np
from plyfile import PlyData

load_dotenv()
UPSAMPLING_URL = os.getenv("UPSAMPLING_ENDPOINT", "")
MAIN_URL = os.getenv("MAIN_MODEL_ENDPOINT", "")

auth_req = google.auth.transport.requests.Request()

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def inference_root():
    """
    Inference endpoint.
    """
    return {"message": "Inference endpoint"}

@router.post("/pointcloud")
async def run_inference_pointcloud(file: UploadFile = File(...), file_format: str = "ply"):
    """
    Run inference on the uploaded image.
    """
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    try:
        creds_token_id = service_account.IDTokenCredentials.from_service_account_file(
            "./secrets/key.json",
            target_audience=MAIN_URL
        )
        creds_token_id.refresh(auth_req)
        TOKEN_ID = creds_token_id.token
        
        image_bytes = await file.read()
        input_np = Preprocessor().process(image_bytes)
        payload = {
            "body": json.dumps(input_np.tolist())
        }
              
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN_ID}"
        }
        print("-----Inferencing...-----")
        response = requests.post(f"{MAIN_URL}/predict", headers=headers, json=payload)
        data = response.json()
        
        pointcloud = data.get("point_cloud", [])   
        if not pointcloud:
            raise HTTPException(status_code=500, detail="No pointcloud in response")
        
        gcs = GCSHandler()
        
        file_path = None
        if file_format == "xyz":
            local_path, filename = save_xyz_file(pointcloud)
            file_path = f"prediction_history/xyz/{filename}"
            download_url = gcs.upload_file(local_path, file_path)
        elif file_format == "ply":
            local_path, filename = save_ply_file(pointcloud)
            file_path = f"prediction_history/ply/{filename}"
            download_url = gcs.upload_file(local_path,file_path)            
        
        # test local
        
        # download_url = ""
        # pointcloud = np.loadtxt("./data/predicted_cloud.xyz").tolist()
        return {
            "download_url": download_url,
            "file_path": file_path, 
            "pointcloud_data": pointcloud,
            "message": "Inference completed successfully"
        }

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/upsampling")
async def run_upsampling_pointcloud(request: RequestUpsampling):
    try: 
        if not request.file_path:
            raise HTTPException(status_code=400, detail="File path is required")
        
        creds_token_id = service_account.IDTokenCredentials.from_service_account_file(
            "./secrets/key.json",
            target_audience=UPSAMPLING_URL
        )
        creds_token_id.refresh(auth_req)
        TOKEN_ID = creds_token_id.token

        file_path = request.file_path
        file_format = request.file_format
        if file_path:
            gcs = GCSHandler()
            
            file_extension = os.path.splitext(file_path)[1]
            tmp_file = NamedTemporaryFile(delete=False, suffix=file_extension)
            local_path, blob = gcs.download_file(file_path, tmp_file.name)
            
            if file_extension == '.xyz':
                input_np = np.loadtxt(local_path)
            elif file_extension == '.ply':
                plydata = PlyData.read(local_path)
                vertex = plydata['vertex']
                input_np = np.column_stack((vertex['x'], vertex['y'], vertex['z']))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
            payload = {
                "instances": [
                    {
                        "body": json.dumps(input_np.tolist())
                    }
                ]
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN_ID}"
            }
            response = requests.post(f"{UPSAMPLING_URL}/predict", headers=headers, json=payload)
            
            data = response.json()
            output = data.get("predictions")[0].get("point_cloud")
            
            if not output: 
                raise HTTPException(status_code=500, detail="No point cloud data in response")
            file_path = None
            if file_format == "xyz":
                local_path, filename = save_xyz_file(output)
                file_path = f"prediction_history/xyz/{filename}"
                download_url = gcs.upload_file(local_path, file_path)
            elif file_format == "ply":
                local_path, filename = save_ply_file(output)
                file_path = f"prediction_history/ply/{filename}"
                download_url = gcs.upload_file(local_path, file_path)
            return {
                "download_url": download_url,
                "file_path": file_path,
                "pointcloud_data": output,
                "message": "Upsampling completed successfully"
            }
        
        #  test local
        # file_path = "./data/output_pointcloud_8192.ply"
        # plydata = PlyData.read(file_path)
        # vertex = plydata['vertex']
        # input_np = np.column_stack((vertex['x'], vertex['y'], vertex['z']))
        # return {
        #     "download_url": "",
        #     "pointcloud_data": input_np.tolist(),
        #     "message": "Upsampling completed successfully"
        # }
    except Exception as e:
        print(f"Error during upsampling: {e}")
        raise HTTPException(status_code=500, detail=str(e))