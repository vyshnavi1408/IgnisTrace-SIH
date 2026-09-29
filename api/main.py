import os
import json
import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="IgnisTrace API",
    description="Backend for industrial fire and persistent thermal source detection",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EVENTS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "features.geojson"
)

def clean_nan_values(obj):
    if isinstance(obj,dict):
        return{
            key:clean_nan_values(value)
            for key,value in obj.items() 

        }
    if isinstance(obj,list):
        return[
            clean_nan_values(value)
            for value in obj
        ]
    if isinstance(obj,float) and not math.isfinite(obj):
        return None
    return obj



@app.get("/")
def root():
    return {
        "message": "IgnisTrace API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
allow_origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5173",
    "http://localhost:5173"
],

@app.get("/events")
def get_events():

    if not os.path.exists(EVENTS_PATH):
        return {
            "error": "events.geojson not found"
        }

    with open(EVENTS_PATH, "r") as f:
        data = json.load(f)

        data = clean_nan_values(data)

    return data
