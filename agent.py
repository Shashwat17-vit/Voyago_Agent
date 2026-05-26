import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

load_dotenv()

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=8192
)

tavily = TavilySearch(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# --- State ---
class ItineraryState(TypedDict):
    trip: dict
    preferences: dict
    search_context: str
    itinerary: dict

# --- Node 1: search Tavily, keep only short snippets ---
def search_node(state: ItineraryState) -> ItineraryState:
    trip = state["trip"]
    prefs = state["preferences"]
    query = f"top attractions restaurants things to do {trip['destination']} {prefs['interests']}"

    results = tavily.invoke(query)

    snippets = []
    if isinstance(results, str):
        snippets.append(results[:600])
    elif isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                title = r.get("title", "")
                snippet = r.get("content", r.get("snippet", ""))[:200]
                snippets.append(f"- {title}: {snippet}")
            elif isinstance(r, str):
                snippets.append(f"- {r[:200]}")

    return {"search_context": "\n".join(snippets)}

# --- Node 2: generate itinerary using condensed search context ---
def generate_node(state: ItineraryState) -> ItineraryState:
    trip = state["trip"]
    prefs = state["preferences"]
    context = state["search_context"]

    prompt = f"""You are a travel planner. Create a day-by-day itinerary for this trip.

Destination: {trip['destination']}
Dates: {trip['startDate']} to {trip['endDate']}
Travelers: {trip['numTravelers']}
From: {prefs['currentLocation']}
Budget: {prefs['budget']}
Type: {prefs['tripType']}
Accommodation: {prefs['accommodation']}
Transportation: {prefs['transportation']}
Interests: {prefs['interests']}
Notes: {prefs.get('notes', '')}

Real highlights found:
{context}

Return ONLY valid JSON, no markdown:
{{"days":[{{"dayNumber":1,"dayLabel":"Day 1 — City","date":"YYYY-MM-DD","events":[{{"title":"","description":"","locationName":"","latitude":0.0,"longitude":0.0,"category":"SIGHTSEEING","startTime":"09:00","endTime":"11:00","orderIndex":1}}]}}]}}
Categories: SIGHTSEEING, FOOD, ACTIVITY, TRANSPORT, ACCOMMODATION. 3-4 events per day."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"itinerary": extract_json(response.content)}

# --- Build graph: search → generate ---
builder = StateGraph(ItineraryState)
builder.add_node("search", search_node)
builder.add_node("generate", generate_node)
builder.set_entry_point("search")
builder.add_edge("search", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# --- Helpers ---
import re
import logging

logger = logging.getLogger(__name__)

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the outermost JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Attempt to fix truncated JSON by closing open brackets
        fixed = candidate
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        # Remove trailing comma before closing
        fixed = re.sub(r',\s*$', '', fixed)
        fixed += ']' * open_brackets + '}' * open_braces
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.error("JSON repair failed: %s\nRaw text (last 500 chars): ...%s", e, text[-500:])
            raise

    raise json.JSONDecodeError("No JSON object found in LLM response", text, 0)

def mock_itinerary(trip: dict) -> dict:
    return {
        "days": [
            {
                "dayNumber": 1,
                "dayLabel": f"Day 1 — {trip['destination']}",
                "date": trip['startDate'],
                "events": [
                    {
                        "title": "Arrival & Hotel Check-in",
                        "description": "Arrive and settle into your accommodation.",
                        "locationName": f"Hotel, {trip['destination']}",
                        "latitude": 48.8566,
                        "longitude": 2.3522,
                        "category": "ACCOMMODATION",
                        "startTime": "14:00",
                        "endTime": "15:00",
                        "orderIndex": 1
                    },
                    {
                        "title": "Evening Dinner",
                        "description": "Explore local cuisine.",
                        "locationName": f"Restaurant, {trip['destination']}",
                        "latitude": 48.8606,
                        "longitude": 2.3376,
                        "category": "FOOD",
                        "startTime": "19:00",
                        "endTime": "21:00",
                        "orderIndex": 2
                    }
                ]
            }
        ]
    }

# --- Entry point ---
async def generate_itinerary(trip: dict, preferences: dict) -> dict:
    if USE_MOCK:
        return mock_itinerary(trip)

    result = graph.invoke({
        "trip": trip,
        "preferences": preferences,
        "search_context": "",
        "itinerary": {}
    })
    return result["itinerary"]
