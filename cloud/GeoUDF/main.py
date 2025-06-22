import os
import uuid
import traceback

from fastapi import FastAPI, HTTPException, Request
import torch
from torch import nn
import pymeshlab
import numpy as np
import trimesh

from model import PUGeo, UDF
from utils import download_from_gcs, get_nn_dist, get_udf, custom_marching_cube, upload_to_gcs

app = FastAPI()

lambda_1 = 100
lambda_2 = 1
lambda_3 = 0.1

is_scale = True # whether scale the input into a unit cube

weights_path = "weights"
pu_model = None
udf_model = None

@app.on_event("startup")
def startup_event():
    global pu_model, udf_model
    try:
        pu_model = PUGeo(knn=20)
        pu_model = nn.DataParallel(pu_model)
        pu_model = pu_model.cuda()
        pu_model.load_state_dict(torch.load(os.path.join(weights_path,'pu_model_best.t7')))
        print("[INFO] PU model loaded successfully.")

        udf_model = UDF()
        udf_model = nn.DataParallel(udf_model)
        udf_model = udf_model.cuda()
        udf_model.load_state_dict(torch.load(os.path.join(weights_path,'udf_model_best.t7')))
        print("[INFO] UDF model loaded successfully.")

        pu_model.eval()
        udf_model.eval()
        print("[INFO] ✅ Initialization complete.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    try:
        body = await request.json()
        print(f"[INFO] Receiving prediction request... : {body}")
        file_path = body.get("file_path")
        res = body.get("res", 128)  # default 128, options are [128, 192]
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path is required")
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension != '.ply':
            raise HTTPException(status_code=400, detail="Only .ply files are supported")
        input_path = download_from_gcs(file_path, destination_path="temp")

        ms_set = pymeshlab.MeshSet()
        ms_set.load_new_mesh(input_path)
        ms_set.compute_normal_for_point_clouds()

        sparse_pc=np.array(ms_set.current_mesh().vertex_matrix())
        normals=np.array(ms_set.current_mesh().vertex_normal_matrix())

        if is_scale:
            sparse_pc_max=np.max(sparse_pc,axis=0,keepdims=True)
            sparse_pc_min=np.min(sparse_pc,axis=0,keepdims=True)

            center=(sparse_pc_max+sparse_pc_min)/2
            scale=np.max(sparse_pc_max-sparse_pc_min)

            sparse_pc=(sparse_pc-center)/scale

        # output_dict['dense_xyz'].detach()           #(1,3,N)
        dense_pc=  torch.from_numpy(sparse_pc).unsqueeze(0).float().cuda()  
        # output_dict['dense_normal'].detach()    #(1,3,N)
        dense_normal=torch.from_numpy(normals).unsqueeze(0).float().cuda()

        output_dict={}
        output_dict['dense_xyz']=dense_pc
        output_dict['dense_normal']=dense_normal

        max_batch=2**16

        N = res
        size=1.05
        voxel_size=size/(N-1)

        edge_interp_vert={}
        vert_list=[]
        face_list=[]

        grids_verts=np.mgrid[:N,:N,:N]
        grids_verts=np.moveaxis(grids_verts,0,-1)   
        

        grids_coords=grids_verts/(N-1)*size-size/2

        grids_coords_flatten=np.asarray(grids_coords.reshape(-1,3),dtype=np.float64)    #(N**3, 3)

        grids_udf_flatten=np.zeros((N**3,))
        grids_udf_grad_flatten=np.zeros((N**3,3))

        num_samples=N**3

        head=0

        while head<num_samples:
            sample_subset=torch.from_numpy(grids_coords_flatten[head:min(head+max_batch,num_samples),:]).cuda().float()
            with torch.no_grad():
                df=get_nn_dist(dense_pc.squeeze(0).reshape(-1,3),sample_subset)
            grids_udf_flatten[head:min(head+max_batch,num_samples)]=df.detach().cpu().numpy()
            head=head+max_batch

        norm_mask=grids_udf_flatten<voxel_size*2
        norm_idx=np.where(norm_mask)[0]
        head,num_samples=0,norm_idx.shape[0]

        while head < num_samples:
            sample_subset_mask=np.zeros_like(norm_mask)
            sample_subset_mask[norm_idx[head:min(head+max_batch,num_samples)]]=True
            sample_subset=torch.from_numpy(grids_coords_flatten[sample_subset_mask, :]).cuda().float()

            with torch.no_grad():
                df,df_grad=get_udf(udf_model,output_dict,sample_subset)

            grids_udf_flatten[sample_subset_mask]=df.detach().cpu().numpy()
            grids_udf_grad_flatten[sample_subset_mask,:]=df_grad.detach().cpu().numpy()

            head=head+max_batch
        
        grids_udf=grids_udf_flatten.reshape(N,N,N)
        grids_udf_grad=grids_udf_grad_flatten.reshape(N,N,N,3)

        vs,fs=custom_marching_cube(grids_coords,grids_udf,grids_udf_grad,voxel_size,N)

        if is_scale:
            vs = vs * scale + center
        
        mesh=trimesh.Trimesh(vs,fs)

        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.fill_holes()
        
        temp_filename = f"{uuid.uuid4()}.ply"
        mesh.export(temp_filename)
        blob_path = f"prediction_history/ply/{os.path.basename(temp_filename)}"
        uploaded_filepath = upload_to_gcs(temp_filename, blob_path)
        
        os.remove(temp_filename)
        
        torch.cuda.empty_cache()
        
        return {"file_path": uploaded_filepath}
            
    except ValueError:
        print("[ERROR] Invalid JSON format")
        stacktrace = traceback.format_exc()
        print(stacktrace)
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "Invalid JSON format",
                "stacktrace": stacktrace
            }
        )
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        stacktrace = traceback.format_exc()
        print(stacktrace)
        raise HTTPException(
            status_code=500, 
            detail={
                "error": f"Unexpected error: {str(e)}",
                "stacktrace": stacktrace
            }
        )
