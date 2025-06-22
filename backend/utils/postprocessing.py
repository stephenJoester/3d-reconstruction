from tempfile import NamedTemporaryFile
import os
import open3d as o3d
import numpy as np

def save_xyz_file(pointcloud_data):
    """
    Save the point cloud data to a .xyz file.
    """
    # Define the path to save the .xyz file
    tmp_file = NamedTemporaryFile(delete=False, suffix=".xyz", mode='w')
    for point in pointcloud_data:
        if isinstance(point ,list) and len(point) >= 3:
            x, y, z = point[:3]
            tmp_file.write(f"{x} {y} {z}\n")
    tmp_file.close()
    return tmp_file.name, os.path.basename(tmp_file.name)

def save_ply_file(pointcloud_data): 
    """
    Save the point cloud data to a .ply file.
    """
    points = np.array(pointcloud_data, dtype=np.float32)
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    tmp_file = NamedTemporaryFile(delete=False, suffix=".ply")
    o3d.io.write_point_cloud(tmp_file.name, point_cloud)
    
    return tmp_file.name, os.path.basename(tmp_file.name)

def taubin_smoothing(input_path, output_path, iterations=20, lambda_val=0.5, mu_val=-0.53):
    mesh = o3d.io.read_triangle_mesh(input_path)
    if len(mesh.vertices) == 0 or np.any(np.isnan(np.asarray(mesh.vertices))):
        raise ValueError("Input mesh is empty or contains NaN values.")
    
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    smoothed_mesh = mesh.filter_smooth_taubin(
        number_of_iterations=iterations,
        lambda_filter=lambda_val,
        mu=mu_val
    )
    
    smoothed_vertices = np.asarray(smoothed_mesh.vertices)
    if len(smoothed_vertices) == 0:
        raise ValueError("Smoothed mesh has no vertices.")
    if np.any(np.isnan(smoothed_vertices)) or np.any(np.isinf(smoothed_vertices)):
        raise ValueError("Smoothed mesh contains NaN or Inf values.")
    
    smoothed_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_path, smoothed_mesh)
    
def laplacian_smoothing(input_path, output_path, iterations=20, lambda_val=0.5):
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    if len(mesh.vertices) == 0 or np.any(np.isnan(np.asarray(mesh.vertices))):
        raise ValueError("Input mesh is empty or contains NaN values.")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    smoothed_mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=iterations,
        lambda_filter=lambda_val
    )

    smoothed_vertices = np.asarray(smoothed_mesh.vertices)
    if np.any(np.isnan(smoothed_vertices)) or np.any(np.isinf(smoothed_vertices)):
        raise ValueError("Smoothed mesh contains NaN or Inf values.")

    smoothed_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_path, smoothed_mesh)
    