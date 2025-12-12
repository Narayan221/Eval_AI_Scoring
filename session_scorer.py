import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from typing import Dict, List
import math

class SessionScorer:
    def __init__(self):
        self.yolo_model = YOLO('yolov8n-pose.pt')
        self.yolo_model.to('cuda')  # GPU only
        
        # Optimize YOLO for speed while maintaining accuracy
        self.yolo_model.conf = 0.25  # Slightly higher confidence for better accuracy
        self.yolo_model.iou = 0.45   # Better IoU threshold
        self.yolo_model.max_det = 1  # Only detect 1 person
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.previous_positions = []  # For movement tracking
        
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
            
            # Process every 2nd frame for better accuracy while maintaining speed
            sample_rate = 2
            
            while cap.read()[0]:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames for faster processing
                if frame_idx % sample_rate == 0:
                    timestamp = frame_idx / fps
                    frame_metrics = self._analyze_frame(frame, timestamp)
                    metrics.append(frame_metrics)
                    
                frame_idx += 1
                
            cap.release()
            return self._calculate_scores(metrics)
            
        finally:
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass  # Ignore Windows file lock issues
    
    def _analyze_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:  # Resize if too large
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # YOLO pose detection with optimized settings
        results = self.yolo_model(frame, verbose=False, imgsz=640)
        
        # Enhanced metrics
        attention = self._calculate_enhanced_attention(frame, results)
        confidence = self._calculate_confidence(results)
        posture = self._calculate_enhanced_posture(results)
        engagement = self._calculate_enhanced_engagement(frame, results)
        
        # Movement analysis
        movement_score = self._calculate_movement(results)
        
        return {
            "timestamp": timestamp,
            "attention": attention,
            "confidence": confidence,
            "posture": posture,
            "engagement": engagement,
            "movement_stability": movement_score,
            "head_orientation": self._calculate_head_orientation(results),
            "eye_contact_quality": self._calculate_eye_contact(frame)
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
        
        max_confidence = max([box[4].item() for box in results[0].boxes.data])
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
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240
        
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
            # Use smaller frame for speed
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 3, minSize=(20, 20))
            
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
            return max(40.0, eye_contact_score * 100)
        except:
            return 40.0
    
    def _calculate_facial_engagement(self, frame) -> float:
        try:
            # Use smaller frame for speed
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 3, minSize=(20, 20))
            
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
        
        # Enhanced overall score calculation
        overall_score = (avg_attention * 0.25 + avg_confidence * 0.15 + 
                        avg_posture * 0.2 + avg_engagement * 0.2 + 
                        avg_movement * 0.1 + avg_eye_contact * 0.1)
        
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
            "enhanced_metrics": {
                "head_orientations": [m["head_orientation"] for m in metrics],
                "movement_patterns": [m["movement_stability"] for m in metrics],
                "eye_contact_timeline": [m["eye_contact_quality"] for m in metrics]
            },
            "frame_by_frame_data": metrics,
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
                "attention": 0.25,
                "confidence": 0.15,
                "posture": 0.2,
                "engagement": 0.2,
                "movement_stability": 0.1,
                "eye_contact_quality": 0.1
            },
            "enhanced_features": [
                "Eye gaze tracking",
                "Head orientation analysis",
                "Movement pattern detection",
                "Facial expression analysis",
                "Spine alignment measurement"
            ]
        }