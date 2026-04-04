# OrthoVision

OrthoVision is an AI-assisted fracture analysis web application.
It combines a Flask backend and a React (Vite) frontend to:

- analyze uploaded X-ray images,
- predict fracture presence and likely fracture type,
- generate Grad-CAM heatmaps,
- persist user and analysis history in MongoDB.

## Features

- X-ray image upload and fracture inference
- Confidence score and fracture-type prediction
- Heatmap generation (`/get_heatmap`) for visual explainability
- User profile storage and analysis history endpoints
- SPA frontend with analysis dashboard and 3D-oriented UI components

## Tech Stack

- Backend: Flask, TensorFlow/Keras, OpenCV, PyMongo
- Frontend: React 19, Vite, Axios, React Router
- Database: MongoDB
- Model files: Keras `.keras` models in `backend/models/`

## Project Structure

```text
OrthoVision/
  backend/
    app.py
    database.py
    utils.py
    models/
  frontend/
    src/
    public/
    package.json
  dataset/
  train_fracture_type.py
  requirements.txt
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB (local or remote URI)

## Setup

### 1. Clone and enter the project

```powershell
git clone https://github.com/nethmitharushika56/OrthoVision.git
cd OrthoVision
```

### 2. Create and activate Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```powershell
npm --prefix frontend install
```

## Environment Variables

The backend works without extra env vars, but you can override defaults:

- `MONGODB_URI` (default: `mongodb://localhost:27017`)
- `MONGODB_DB_NAME` (default: `orthovision`)
- `FRACTURE_THRESHOLD` (default: `0.60`)
- `TYPE_MIN_CONF` (default: `0.50`)
- `TYPE_MIN_MARGIN` (default: `0.08`)
- `BINARY_MIN_MARGIN` (default: `0.08`)
- `MIN_LOCALIZATION_CONF` (default: `0.25`)
- `MIN_LOCALIZATION_QUALITY` (default: `0.15`)
- `MIN_HEATMAP_CONF` (default: `0.20`)
- `MIN_HEATMAP_MARGIN` (default: `0.03`)

Example (PowerShell):

```powershell
$env:MONGODB_URI="mongodb://localhost:27017"
$env:MONGODB_DB_NAME="orthovision"
```

## Running Locally (Development)

### Terminal 1: Start backend

```powershell
.\.venv\Scripts\Activate.ps1
python backend/app.py
```

Backend runs on `http://localhost:5000`.

### Terminal 2: Start frontend

```powershell
npm --prefix frontend run dev -- --host
```

Frontend runs on Vite default port (`5173`, or next available).

The Vite config proxies API routes to the backend, so frontend calls can use relative paths such as `/analyze` and `/get_heatmap`.

## Running with VS Code Tasks

This workspace includes tasks:

- `run-backend`
- `run-frontend-dev`
- `build-frontend`

You can run them from VS Code: `Terminal -> Run Task...`.

## Production Build (Frontend)

```powershell
npm --prefix frontend run build
```

The Flask app is configured to serve the built frontend from `frontend/dist`.

## API Endpoints

### Health

- `GET /health` - backend/model/db status
- `GET /db/health` - database connectivity

### Users

- `POST /users/upsert` - create or update a user
- `GET /users/<email>` - fetch user by email

### Analyses

- `POST /analyses` - store analysis record
- `GET /analyses?userEmail=<email>` - list analyses

### Inference and Files

- `POST /analyze` - upload image and run prediction
  - multipart field: `image`
  - optional form field: `userEmail`
- `GET /get_heatmap` - latest generated heatmap image
- `GET /image/<filename>` - uploaded image retrieval

## Model Files

Expected model artifacts:

- `backend/models/orthovision_model.keras` (binary fracture model)
- `backend/models/orthovision_type_model.keras` (fracture type model)
- `backend/models/orthovision_model.class_indices.json`
- `backend/models/orthovision_type_model.class_indices.json`

If the primary model is missing, backend logic can fall back to the type model when available.

## Training

A fracture type training script is included:

```powershell
.\.venv\Scripts\Activate.ps1
python train_fracture_type.py
```

Dataset layout expected:

```text
dataset/<Fracture Type>/Train/*.jpg|*.png
dataset/<Fracture Type>/Test/*.jpg|*.png
```

Outputs:

- `backend/models/orthovision_type_model.keras`
- `backend/models/orthovision_type_model.class_indices.json`

For additional training notes, see `IMPROVED_TRAINING_GUIDE.md`.

## Troubleshooting

- Frontend cannot connect: ensure backend is running on port `5000`.
- Vite port mismatch: if `5173` is busy, Vite may start on `5174`.
- Heatmap not found: call `POST /analyze` first, then `GET /get_heatmap`.
- MongoDB errors: verify `MONGODB_URI` and local MongoDB service status.
- Model file missing: place a valid `.keras` model inside `backend/models/`.

## License

This project is licensed under the terms in `LICENSE`.
