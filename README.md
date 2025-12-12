# Session Scoring API

FastAPI project for analyzing user engagement from video using YOLO models.

## Setup
```bash
pip install -r requirements.txt
python run.py
```

## API Endpoints

### POST /analyze-session
Upload video file to get session scoring analysis.

### GET /scoring-formula
Get information about scoring formulas and weights.

## Scoring Formula
**Overall Score = 0.3 × Attention + 0.2 × Confidence + 0.25 × Posture + 0.25 × Engagement**

- **Attention**: Face position relative to frame center (0-100)
- **Confidence**: YOLO person detection confidence (0-100)  
- **Posture**: Shoulder alignment measurement (0-100)
- **Engagement**: Body presence and visibility in frame (0-100)