from fastapi import FastAPI, HTTPException, Request
import torch
import numpy as np
import json
import os
from sklearn.neighbors import KDTree
from tqdm import tqdm

import fn_config, fd_config
from utils import farthest_point_sample, rotation_matrix_from_vectors
app = FastAPI()

tpointnumber = 8192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "./model-store"
model_fn = None
model_fd = None
process_folder = 'process/upsampling'
bucket_name = '3d-reconstruction-bucket'

@app.on_event("startup")
async def startup_event():
    global model_fn, model_fd
    
    print("[INFO] Loading models...")
    model_fn = fn_config.get_model(device)
    model_fd = fd_config.get_model(device)
    
    print(f"[INFO] Loading model weights from ./combined_model.pt")
    combined_state_dict = torch.load(f"{model_dir}/combined_model.pt", map_location=device)
    model_fn.load_state_dict(combined_state_dict['model1'])
    model_fd.load_state_dict(combined_state_dict['model2'])
    
    model_fn.eval()
    model_fd.eval()
    print("[INFO] ✅ Initialization complete.")
    
@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    try: 
        body = await request.json()
        instances = body["instances"]
        json_str = instances[0]["body"]
        
        xyz_data = np.asarray(json.loads(json_str))
        cloud = xyz_data[:, 0:3]
        
        # Normalize
        bbox=np.zeros((2,3))
        bbox[0][0]=np.min(cloud[:,0])
        bbox[0][1]=np.min(cloud[:,1])
        bbox[0][2]=np.min(cloud[:,2])
        bbox[1][0]=np.max(cloud[:,0])
        bbox[1][1]=np.max(cloud[:,1])
        bbox[1][2]=np.max(cloud[:,2])
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()
        scale1 = 1/scale
        for i in range(cloud.shape[0]):
            cloud[i]=cloud[i]-loc
            cloud[i]=cloud[i]*scale1
        # save to file
        print("Saving point cloud to test.xyz")
        np.savetxt("test.xyz", cloud)
        cloud = np.expand_dims(cloud, 0)
    
        # Inference
        data = np.squeeze(cloud, 0)
        tree1 = KDTree(data)
        
        # generate seed points
        print("Generating seed points")
        command = f"./dense 0.004 {data.shape[0]}"
        print(f"Running command: {command}")
        os.system(command)
        
        data2 = np.loadtxt("./target.xyz")
        xyz2 = data2[:, 0:3]
        pp = xyz2.shape[0] // 400
        p_split = np.array_split(xyz2, pp, axis=0)
        
        print("fn")
        normal = None
        
        for i in tqdm(range(len(p_split))):
            dist, idx = tree1.query(p_split[i], 100)
            cloud = data[idx]
            cloud = cloud - np.tile(np.expand_dims(p_split[i], 1), (1, 100, 1))
            
            with torch.no_grad():
                c = model_fn.encode_inputs(torch.from_numpy(np.expand_dims(cloud, 0)).float().to(device))
            with torch.no_grad():
                n = model_fn.decode(c)
            
            n = n.detach().cpu().numpy()
            if normal is None:
                normal = n
            else: 
                normal = np.append(normal, n, axis=0)
        n_split = np.array_split(normal, pp, axis=0)
        xyzout = []
            
        print("fd")
        for i in tqdm(range(len(n_split))):
            dist, idx = tree1.query(p_split[i], 100)
            cloud = data[idx]
            cloud = cloud - np.tile(np.expand_dims(p_split[i], 1), (1, 100, 1))
            for j in range(cloud.shape[0]):
                M1 = rotation_matrix_from_vectors(n_split[i][j], [1, 0, 0])
                cloud[j] = (np.matmul(M1, cloud[j].T)).T
            
            with torch.no_grad():
                c = model_fd.encode_inputs(torch.from_numpy(np.expand_dims(cloud, 0)).float().to(device))
            with torch.no_grad():
                n = model_fd.decode(c)
            
            length = np.tile(np.expand_dims(n.detach().cpu().numpy(), 1), (1, 3))
            xyzout.extend((p_split[i] + n_split[i] * length).tolist())
            
        xyzout = np.array(xyzout)
        
        # remove outliers
        print("Remove outliers")
        tree3 = KDTree(xyzout)
        dist, idx = tree3.query(xyzout, 30)
        avg = np.mean(dist, axis=1)
        avg_total = np.mean(dist)
        idx = np.where(avg < avg_total * 1.5)[0]
        xyzout = xyzout[idx, :]
        
        pointcloud = xyzout
        # farthest point sample
        for i in range(pointcloud.shape[0]):
            pointcloud[i]=pointcloud[i]*scale
            pointcloud[i]=pointcloud[i]+loc
        
        centroids = farthest_point_sample(pointcloud, tpointnumber)

        final_output = pointcloud[centroids].tolist()
        return {"predictions": [{"point_cloud": final_output}]}
        
    except Exception as e:
        print("❌ Error in predict():", e)
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))