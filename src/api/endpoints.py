"""
FastAPI endpoints for accessing Virginia Open Data Portal metadata.
"""
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add parent directory to path to import from src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.metadata_extractor import VirginiaDataPortalExtractor

app = FastAPI(
    title="Virginia Open Data Portal API",
    description="API for searching Virginia Open Data Portal metadata using semantic search",
    version="0.1.0"
)

# Initialize the extractor
extractor = VirginiaDataPortalExtractor()

class SearchQuery(BaseModel):
    """Model for search query requests."""
    query: str
    limit: Optional[int] = 5

class SearchResponse(BaseModel):
    """Model for search query responses."""
    results: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Virginia Open Data Portal API", "status": "online"}

@app.post("/search", response_model=SearchResponse)
async def search_datasets(query: SearchQuery):
    """Search for datasets using semantic similarity."""
    try:
        results = extractor.search_similar_datasets(query.query, query.limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    """Start the API server."""
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

if __name__ == "__main__":
    start_server() 