from pydantic import BaseModel, Field

class RequestUpsampling(BaseModel):
    """
    Base request model for inference.
    """
    file_path: str = Field(
        ...,
        description="Path to the point cloud file to be processed. This should be a valid GCS path.",
    )
    file_format: str = Field(
        default="ply",
        description="Format of the point cloud file to be processed. Supported formats: 'xyz', 'ply'.",
    )
    n_points: int = Field(
        default=8192,
        description="Number of points to upsample the point cloud to.",
    )

class RequestMesh(BaseModel):
    """
    Base request model for mesh processing.
    """
    file_path: str = Field(
        ...,
        description="Path to the mesh file to be processed. This should be a valid GCS path.",
    )
    smoothing_algorithm: str = Field(
        default="laplacian",
        description="Smoothing algorithm to apply to the mesh. Supported algorithms: 'laplacian', 'taubin'.",
    )
    smoothing_iterations: int = Field(
        default=10,
        description="Number of iterations for the smoothing algorithm.",
    )