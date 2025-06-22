import traceback
import os
import json

from fastapi import FastAPI, HTTPException, Request
import torch
import numpy as np

from model import Depth2Point

app = FastAPI()

model_path = "model.pt"
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def startup_event():
    global model, device
    try:
        model = Depth2Point(initial_point=0).to(device)
        print("[INFO] Initializing model...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("[INFO] Model loaded successfully.")
        model.eval()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")
    
@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    try: 
        body = await request.json()
        print(f"[INFO] Receiving prediction request... : {body}")
        
        # Extract the input data
        json_data = body.get("body")
        if not json_data:
            raise HTTPException(status_code=400, detail="Invalid input data")
        
        parsed = json.loads(json_data)
        np_array = np.array(parsed)
        input_tensor = torch.tensor(np_array).unsqueeze(0).float().to(device)
        
        # Inference
        with torch.no_grad():
            predicted_point_cloud = model(input_tensor)
            predicted_point_cloud = predicted_point_cloud.view(-1, 3).cpu().numpy().tolist()
        
        return {"point_cloud": predicted_point_cloud}
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction failed")