import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import generate_itinerary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ItineraryRequest(BaseModel):
    trip: dict
    preferences: dict

@app.get("/health")
def health():
    return {"status": "ok", "service": "Voyago Agent"}

@app.post("/generate-itinerary")
async def generate(request: ItineraryRequest):
    try:
        logger.info("Received request: trip=%s, preferences=%s", request.trip, request.preferences)
        result = await generate_itinerary(request.trip, request.preferences)
        return result
    except Exception as e:
        logger.error("Failed to generate itinerary: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
