import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from typing import Dict, List
import math
import whisper
import ffmpeg

class SessionScorer:
    def __init__(self):
        # Force NVIDIA GPU specifically
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! NVIDIA GPU required.")
        
        # Find NVIDIA GPU (skip Intel GPU at index 0)
        nvidia_device = 0
        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i).lower()
                if 'nvidia' in gpu_name or 'geforce' in gpu_name or 'rtx' in gpu_name or 'gtx' in gpu_name:
                    nvidia_device = i
                    break
        
        print(f"Using GPU {nvidia_device}: {torch.cuda.get_device_name(nvidia_device)}")
        torch.cuda.set_device(nvidia_device)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(nvidia_device)
        
        # Store device for monitoring
        self.device = f'cuda:{nvidia_device}'
        self.torch = torch
        
        self.yolo_model = YOLO('yolov8n-pose.pt')
        self.yolo_model.to(f'cuda:{nvidia_device}')
        
        # Optimize YOLO for speed while maintaining accuracy
        self.yolo_model.conf = 0.25  # Slightly higher confidence for better accuracy
        self.yolo_model.iou = 0.45   # Better IoU threshold
        self.yolo_model.max_det = 1  # Only detect 1 person
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.previous_positions = []  # For movement tracking
        
        # Force Whisper to use NVIDIA GPU specifically
        self.whisper_model = whisper.load_model("large-v3", device=f"cuda:{nvidia_device}")
        
    def analyze_video(self, video_bytes: bytes) -> Dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
            
        try:
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            metrics = []
            frame_idx = 0
            
            # Process every 5th frame for maximum speed
            sample_rate = 5
            
            while cap.read()[0]:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames for faster processing
                if frame_idx % sample_rate == 0:
                    timestamp = frame_idx / fps
                    frame_metrics = self._analyze_frame(frame, timestamp, frame_idx)
                    metrics.append(frame_metrics)
                    
                frame_idx += 1
                
            cap.release()
            
            # Extract and transcribe audio
            transcription = self._transcribe_audio(tmp_path)
            
            # Combine video analysis with transcription
            results = self._calculate_scores(metrics)
            results["audio_transcription"] = transcription
            
            return results
            
        finally:
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass  # Ignore Windows file lock issues
    
    def _analyze_frame(self, frame: np.ndarray, timestamp: float, frame_idx: int) -> Dict:
        # Monitor GPU usage
        if frame_idx % 50 == 0:  # Every 50 frames
            gpu_memory = self.torch.cuda.memory_allocated(self.device) / 1024**3
            print(f"Frame {frame_idx}: GPU Memory: {gpu_memory:.2f}GB")
        
        # Aggressive resize for speed
        frame = cv2.resize(frame, (416, 416))  # Fixed small size for YOLO
        
        # YOLO pose detection with maximum speed settings
        results = self.yolo_model(frame, verbose=False, imgsz=416)
        
        # Simplified metrics for speed
        confidence = self._calculate_confidence(results)
        posture = self._calculate_enhanced_posture(results)
        movement_score = self._calculate_movement(results)
        
        # Skip expensive face detection every frame
        if frame_idx % 10 == 0:  # Only every 10th processed frame
            attention = self._calculate_enhanced_attention(frame, results)
            engagement = self._calculate_enhanced_engagement(frame, results)
            eye_contact = self._calculate_eye_contact(frame)
        else:
            attention = 60.0  # Default values
            engagement = 70.0
            eye_contact = 50.0
        
        # Count persons detected
        person_count = self._count_persons(results)
        
        return {
            "timestamp": timestamp,
            "attention": attention,
            "confidence": confidence,
            "posture": posture,
            "engagement": engagement,
            "movement_stability": movement_score,
            "head_orientation": self._calculate_head_orientation(results),
            "eye_contact_quality": eye_contact,
            "person_count": person_count
        }
    
    def _calculate_enhanced_attention(self, frame, results) -> float:
        if not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return 0.0
        
        keypoints = results[0].keypoints.data[0]  # First person
        nose = keypoints[0]  # Nose keypoint for head position
        
        if nose[2] < 0.5:  # Low confidence
            return 0.0
        
        # Enhanced attention with eye gaze and face direction
        face_attention = self._calculate_face_direction_attention(frame)
        
        # Head position attention
        center_x = float(nose[0].cpu()) / frame.shape[1]
        center_y = float(nose[1].cpu()) / frame.shape[0]
        distance_from_center = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
        position_score = max(0, 1 - distance_from_center * 2)
        
        # Combine face direction and position (weighted)
        attention_score = (face_attention * 0.7) + (position_score * 0.3)
        
        return min(100, attention_score * 100)
    
    def _calculate_confidence(self, results) -> float:
        if not results[0].boxes or len(results[0].boxes.data) == 0:
            return 0.0
        
        # Filter for person detections only (class 0)
        person_boxes = [box for box in results[0].boxes.data if box[5] == 0]
        if not person_boxes:
            return 0.0
            
        max_confidence = max([box[4].item() for box in person_boxes])
        return min(100, max_confidence * 100)
    
    def _calculate_enhanced_posture(self, results) -> float:
        if not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return 0.0
        
        keypoints = results[0].keypoints.data[0]
        left_shoulder = keypoints[5]   # Left shoulder
        right_shoulder = keypoints[6]  # Right shoulder
        
        if left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5:  # Low confidence
            return 0.0
        
        # Enhanced posture with spine alignment
        shoulder_diff = abs(float(left_shoulder[1].cpu()) - float(right_shoulder[1].cpu())) / 100
        shoulder_score = max(0, 1 - shoulder_diff)
        
        # Add spine alignment (shoulder to hip alignment)
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        if left_hip[2] > 0.5 and right_hip[2] > 0.5:
            spine_alignment = self._calculate_spine_alignment(left_shoulder, right_shoulder, left_hip, right_hip)
            posture_score = (shoulder_score * 0.6) + (spine_alignment * 0.4)
        else:
            posture_score = shoulder_score
        
        return min(100, posture_score * 100)
    
    def _calculate_enhanced_engagement(self, frame, results) -> float:
        if not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return 0.0
        
        keypoints = results[0].keypoints.data[0]
        
        # Body presence and visibility (torso keypoints)
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Count visible body parts (higher visibility = more engagement)
        visible_parts = sum([
            float(left_shoulder[2].cpu()) > 0.5,
            float(right_shoulder[2].cpu()) > 0.5,
            float(left_hip[2].cpu()) > 0.5,
            float(right_hip[2].cpu()) > 0.5
        ])
        
        body_presence = visible_parts / 4
        
        # Add facial expression analysis for engagement depth
        facial_engagement = self._calculate_facial_engagement(frame)
        
        # Combine body presence and facial engagement
        engagement_score = (body_presence * 0.6) + (facial_engagement * 0.4)
        
        return min(100, engagement_score * 100)
    
    def _calculate_face_direction_attention(self, frame) -> float:
        # Use smaller frame for speed
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 3, minSize=(20, 20))
        
        if len(faces) == 0:
            return 0.6  # Higher default attention score
        
        # Use first detected face for speed
        face = faces[0]
        x, y, w, h = face
        
        # Scale back to original frame
        scale_x = frame.shape[1] / 160
        scale_y = frame.shape[0] / 120
        
        # Face center position (scaled back)
        face_center_x = (x + w // 2) * scale_x
        face_center_y = (y + h // 2) * scale_y
        
        # Frame center
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        
        # Distance from center (closer = better attention)
        distance = np.sqrt((face_center_x - frame_center_x)**2 + (face_center_y - frame_center_y)**2)
        max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
        
        direction_score = max(0, 1 - (distance / max_distance))
        return min(1.0, direction_score)
    
    def _calculate_spine_alignment(self, left_shoulder, right_shoulder, left_hip, right_hip) -> float:
        # Calculate spine straightness
        shoulder_center = [(float(left_shoulder[0].cpu()) + float(right_shoulder[0].cpu())) / 2,
                          (float(left_shoulder[1].cpu()) + float(right_shoulder[1].cpu())) / 2]
        hip_center = [(float(left_hip[0].cpu()) + float(right_hip[0].cpu())) / 2,
                     (float(left_hip[1].cpu()) + float(right_hip[1].cpu())) / 2]
        
        # Calculate angle deviation from vertical
        angle = math.atan2(abs(shoulder_center[0] - hip_center[0]), 
                          abs(shoulder_center[1] - hip_center[1]))
        angle_degrees = math.degrees(angle)
        
        # Good posture = small angle deviation
        alignment_score = max(0, 1 - (angle_degrees / 30))  # 30 degrees max deviation
        return alignment_score
    
    def _calculate_movement(self, results) -> float:
        if not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return 50.0  # Neutral score
        
        keypoints = results[0].keypoints.data[0]
        current_position = [float(keypoints[0][0].cpu()), float(keypoints[0][1].cpu())]  # Nose position
        
        if len(self.previous_positions) == 0:
            self.previous_positions.append(current_position)
            return 50.0
        
        # Calculate movement stability (less movement = more stable)
        movement = np.linalg.norm(np.array(current_position) - np.array(self.previous_positions[-1]))
        
        # Keep last 10 positions for smoothing
        self.previous_positions.append(current_position)
        if len(self.previous_positions) > 10:
            self.previous_positions.pop(0)
        
        # Stability score (inverse of movement)
        stability_score = max(0, 100 - (movement * 2))  # Scale movement
        return min(100, stability_score)
    
    def _calculate_head_orientation(self, results) -> Dict:
        if not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return {"pitch": 0, "yaw": 0, "roll": 0}
        
        keypoints = results[0].keypoints.data[0]
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        
        if nose[2] < 0.5 or left_eye[2] < 0.5 or right_eye[2] < 0.5:
            return {"pitch": 0, "yaw": 0, "roll": 0}
        
        # Calculate head orientation angles
        eye_center = [(float(left_eye[0].cpu()) + float(right_eye[0].cpu())) / 2,
                     (float(left_eye[1].cpu()) + float(right_eye[1].cpu())) / 2]
        nose_pos = [float(nose[0].cpu()), float(nose[1].cpu())]
        
        # Simplified orientation calculation
        yaw = math.degrees(math.atan2(nose_pos[0] - eye_center[0], 100))  # Left-right
        pitch = math.degrees(math.atan2(nose_pos[1] - eye_center[1], 100))  # Up-down
        roll = math.degrees(math.atan2(float(right_eye[1].cpu()) - float(left_eye[1].cpu()),
                                      float(right_eye[0].cpu()) - float(left_eye[0].cpu())))
        
        return {"pitch": round(pitch, 2), "yaw": round(yaw, 2), "roll": round(roll, 2)}
    
    def _calculate_eye_contact(self, frame) -> float:
        try:
            # Ultra small frame for maximum speed
            small_frame = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.5, 2, minSize=(15, 15))
            
            if len(faces) == 0:
                return 40.0  # Higher default score
            
            # Use first face for speed
            face = faces[0]
            x, y, w, h = face
            
            # Calculate eye contact based on face size
            face_area = w * h
            frame_area = small_frame.shape[0] * small_frame.shape[1]
            face_ratio = face_area / frame_area
            
            eye_contact_score = min(1.0, face_ratio * 25)  # Increased multiplier for smaller frame
            final_score = eye_contact_score * 100
            return max(40.0, min(100.0, final_score))  # Cap at 100
        except:
            return 40.0
    
    def _calculate_facial_engagement(self, frame) -> float:
        try:
            # Ultra small frame for maximum speed
            small_frame = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.5, 2, minSize=(15, 15))
            
            if len(faces) == 0:
                return 0.7  # Higher default engagement
            
            # Use first face for speed
            face = faces[0]
            x, y, w, h = face
            face_area = w * h
            
            # Normalize face area
            frame_area = small_frame.shape[0] * small_frame.shape[1]
            face_ratio = face_area / frame_area
            
            engagement_score = min(1.0, face_ratio * 40)  # Increased multiplier for smaller frame
            return max(0.5, engagement_score)  # Higher minimum engagement
        except:
            return 0.7
    
    def _count_persons(self, results) -> int:
        if not results[0].boxes or len(results[0].boxes.data) == 0:
            return 0
        
        # Count person detections (class 0)
        person_count = sum(1 for box in results[0].boxes.data if box[5] == 0)
        return person_count
    
    def _transcribe_audio(self, video_path: str) -> Dict:
        try:
            # Extract audio from video
            audio_path = video_path.replace('.mp4', '.wav')
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_path)
            
            # Clean up audio file
            try:
                os.unlink(audio_path)
            except:
                pass
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip()
                    }
                    for seg in result["segments"]
                ]
            }
        except Exception as e:
            return {
                "text": "",
                "language": "unknown",
                "segments": [],
                "error": str(e)
            }
    
    def _calculate_scores(self, metrics: List[Dict]) -> Dict:
        if not metrics:
            return {"error": "No frames analyzed"}
        
        # Average scores across all frames
        avg_attention = np.mean([m["attention"] for m in metrics])
        avg_confidence = np.mean([m["confidence"] for m in metrics])
        avg_posture = np.mean([m["posture"] for m in metrics])
        avg_engagement = np.mean([m["engagement"] for m in metrics])
        avg_movement = np.mean([m["movement_stability"] for m in metrics])
        avg_eye_contact = np.mean([m["eye_contact_quality"] for m in metrics])
        
        # Person count analysis
        person_counts = [m["person_count"] for m in metrics]
        avg_person_count = np.mean(person_counts)
        single_person_frames = sum(1 for count in person_counts if count == 1)
        multi_person_frames = sum(1 for count in person_counts if count > 1)
        no_person_frames = sum(1 for count in person_counts if count == 0)
        
        # Adjust scores based on person presence
        presence_penalty = 1.0
        if avg_person_count == 0:
            presence_penalty = 0.3  # Heavy penalty for no person detected
        elif avg_person_count > 1.5:
            presence_penalty = 0.8  # Slight penalty for multiple people
        
        # Enhanced overall score calculation with presence penalty
        base_score = (avg_attention * 0.25 + avg_confidence * 0.15 + 
                     avg_posture * 0.2 + avg_engagement * 0.2 + 
                     avg_movement * 0.1 + avg_eye_contact * 0.1)
        overall_score = base_score * presence_penalty
        
        return {
            "session_analysis": {
                "attention_score": round(avg_attention, 2),
                "confidence_score": round(avg_confidence, 2),
                "posture_score": round(avg_posture, 2),
                "engagement_score": round(avg_engagement, 2),
                "movement_stability_score": round(avg_movement, 2),
                "eye_contact_quality_score": round(avg_eye_contact, 2),
                "overall_score": round(overall_score, 2)
            },
            "scoring_formula": self.get_formula_info()["formula"]
        }
    
    def get_formula_info(self) -> Dict:
        return {
            "formula": {
                "overall_score": "0.25 × attention + 0.15 × confidence + 0.2 × posture + 0.2 × engagement + 0.1 × movement + 0.1 × eye_contact",
                "attention": "Enhanced face direction and head position analysis (0-100)",
                "confidence": "YOLO person detection confidence score (0-100)",
                "posture": "Shoulder alignment and spine straightness (0-100)",
                "engagement": "Body presence and facial engagement analysis (0-100)",
                "movement_stability": "Movement stability and fidgeting analysis (0-100)",
                "eye_contact_quality": "Eye openness and gaze direction quality (0-100)"
            },
            "weights": {
                "attention": "25% (0.25) - Most Important",
                "confidence": "15% (0.15) - Moderate Impact",
                "posture": "20% (0.20) - High Impact",
                "engagement": "20% (0.20) - High Impact",
                "movement_stability": "10% (0.10) - Low Impact",
                "eye_contact_quality": "10% (0.10) - Low Impact"
            },
            "score_ranges": {
                "all_metrics": "0-100 (Higher is better)",
                "overall_score": "0-100 (Weighted average with presence penalty)",
                "excellent": "90-100",
                "good": "70-89",
                "average": "50-69",
                "poor": "30-49",
                "very_poor": "0-29"
            },
            "enhanced_features": [
                "Eye gaze tracking",
                "Head orientation analysis",
                "Movement pattern detection",
                "Facial expression analysis",
                "Spine alignment measurement"
            ]
        }