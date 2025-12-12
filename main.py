from fastapi import FastAPI, UploadFile, File
from session_scorer import SessionScorer
import json

app = FastAPI(title="Session Scoring API")
scorer = SessionScorer()

@app.post("/analyze-session")
async def analyze_session(video: UploadFile = File(...)):
    video_bytes = await video.read()
    results = scorer.analyze_video(video_bytes)
    return results

@app.get("/scoring-formula")
def get_scoring_formula():
    return scorer.get_formula_info()