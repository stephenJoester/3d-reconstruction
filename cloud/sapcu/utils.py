import numpy as np
import torch
# from google.cloud import storage

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions
    
def farthest_point_sample(xyz, pointnumber):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N, C = xyz.shape
    torch.seed()
    xyz=torch.from_numpy(xyz).float().to(device)
    centroids = torch.zeros(pointnumber, dtype=torch.long).to(device)

    distance = torch.ones(N).to(device) * 1e32
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
    farthest[0]=N/2
    for i in range(pointnumber):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.detach().cpu().numpy().astype(int)

# def upload_to_gcs(bucket_name, destination_blob_name, source_file_path):
#         """Uploads a file to the bucket."""
#         client = storage.Client()
#         bucket = client.bucket(bucket_name)
#         blob = bucket.blob(destination_blob_name)
#         blob.upload_from_filename(source_file_path)
#         print(f"Uploaded {source_file_path} to gsL//{bucket_name}/{destination_blob_name}")
        
# def download_from_gcs(bucket_name, source_blob_name, destination_file_path):
#     """Download a blob from the bucket."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)

#     blob.download_to_filename(destination_file_path)
#     print(f"Downloaded gs://{bucket_name}/{source_blob_name} to {destination_file_path}")