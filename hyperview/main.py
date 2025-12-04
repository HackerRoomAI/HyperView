from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import uvicorn
import os
from hyperview.database import init_db, get_table, DB_PATH

app = FastAPI(title="HyperView")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image directory (relative to project root)
IMAGE_DIR = Path("data/hyperview_images")

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/points")
async def get_points():
    table = get_table()
    df = table.to_pandas()
    # Drop the vector column (not needed for visualization, and numpy arrays aren't JSON serializable)
    df = df.drop(columns=["vector"], errors="ignore")
    return df.to_dict(orient="records")

@app.get("/api/images/{image_name}")
async def get_image(image_name: str):
    """Serve images from the hyperview_images directory."""
    image_path = IMAGE_DIR / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/jpeg")

@app.get("/api/labels")
async def get_labels():
    """Get label information."""
    import lancedb
    db = lancedb.connect(DB_PATH)
    if "labels" in db.table_names():
        table = db.open_table("labels")
        return table.to_pandas().to_dict(orient="records")
    return []

def start():
    """Entry point for the application script"""
    uvicorn.run("hyperview.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
