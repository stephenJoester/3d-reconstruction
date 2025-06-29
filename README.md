# 3D Reconstruction from 2D Image

This project implements a 3D reconstruction pipeline that generates object point cloud and mesh from a single 2D image, with a modular, multi-stage design for easier control and fine-tuning.

The core contribution lies in the point cloud prediction model, which takes an RGB-D image as input (RGB + estimated depth from MiDaS) to infer the object's 3D structure. This model is a custom CNN architecture inspired by Pixel2Point, adapted and trained specifically for single-view 3D reconstruction.

The output point cloud is further refined through:

Upsampling using the [SAPCU](https://github.com/xnowbzhao/sapcu) model to improve density and continuity,

Mesh reconstruction via the [GeoUDF](https://github.com/rsy6318/GeoUDF) method,

And optional mesh smoothing to enhance visual quality.

The backend is built with FastAPI, and the frontend is based on the open-source Three.js Editor, allowing users to view and interact with the reconstructed mesh in real time.

📦 Dataset (pre-rendered RGB-D images from ShapeNetCore) is available here: 📁 [Google Drive](https://drive.google.com/drive/folders/1uGwH34-tBan44Jrf-XzospmUaXD1Oqxz?usp=sharing)

## How to Run the Project

## Backend (Python + FastAPI)

```bash
# Navigate to backend directory
cd backend
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server
uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend API will be available at http://localhost:3000

## Frontend (Three.js)

Make sure you have Node.js installed.

```bash
# Navigate to frontend directory
cd three.js

# Install dependencies
npm install

# Start the development server
npm run dev
```
## Model Deployment
Each deep learning model (point cloud prediction, upsampling, mesh reconstruction) is containerized with its own Dockerfile. To run the system properly, follow these steps:

1. Build Docker Images
Navigate to each model directory inside the cloud/ folder and build the Docker image:

```bash
# Example for the main model
cd cloud/depth2point
docker build -t depth2point .
```
```bash
# Example for upsampling
cd cloud/sapcu
docker build -t sapcu .
```
```bash
# Example for mesh reconstruction
cd cloud/GeoUDF
docker build -t geoudf .
```
2. Deploy the Containers
You can deploy these models to any cloud platform (e.g. Cloud Run, Azure Container Apps, etc.), or run them locally:

```bash
# Example: run locally on different ports
docker run -d -p 8501:8501 depth2point
docker run -d -p 8502:8502 sapcu
docker run -d -p 8503:8503 geoudf
```
3. Configure API Endpoints
Once your models are deployed and accessible, update the .env file in the backend directory with their respective URLs:
```bash
MAIN_MODEL_ENDPOINT="http://localhost:8501"
UPSAMPLING_ENDPOINT="http://localhost:8502"
MESH_ENDPOINT="http://localhost:8503"
```
After updating, restart the backend server to apply changes.