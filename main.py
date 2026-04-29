from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import generate_itinerary

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
        result = await generate_itinerary(request.trip, request.preferences)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
