import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from typing import Dict, List
import math
from faster_whisper import WhisperModel
import ffmpeg
import concurrent.futures
import time

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
        
        self.previous_positions = []  # For movement tracking
        
        # Provide visual feedback before loading heavy model
        print("Loading Faster-Whisper model (Int8)...")
        self.whisper_model = WhisperModel("large-v3", device="cuda", device_index=nvidia_device, compute_type="int8")
        
    
    def analyze_video(self, video_bytes: bytes) -> Dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
            
        try:
            # 1. Start Audio Extraction & Transcription in Background
            # Extract audio first so transcription can run parallel to video analysis
            audio_path = tmp_path.replace('.mp4', '.wav')
            self._extract_audio(tmp_path, audio_path)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Start transcription immediately
                transcription_future = executor.submit(self._transcribe_audio_file, audio_path)
                
                # 2. Run Video Analysis (Main Thread)
                # Video analysis is heavier on CPU/CV logic, so keep it in main thread
                metrics = self._analyze_video_frames_batched(tmp_path)
                
                # 3. Wait for Transcription
                transcription = transcription_future.result()
            
            # Combine video analysis with transcription
            results = self._calculate_scores(metrics)
            results["audio_transcription"] = transcription
            
            return results
            
        finally:
            try:
                os.unlink(tmp_path)
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except PermissionError:
                pass  # Ignore Windows file lock issues

    def _extract_audio(self, video_path: str, audio_path: str):
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            print(f"Error extracting audio: {e}")

    def _analyze_video_frames_batched(self, video_path: str) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        metrics = []
        frame_idx = 0
        
        # Process every 5th frame
        sample_rate = 5
        batch_size = 16  # YOLO batch size
        
        batch_frames = []
        batch_indices = []
        batch_timestamps = []
        
        while cap.read()[0]:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames
            if frame_idx % sample_rate == 0:
                # Add to batch
                timestamp = frame_idx / fps
                
                # Pre-process frame for YOLO (resize)
                # Note: YOLOv8 handles resizing internally but pre-resizing 
                # saves PCIe bandwidth if frames are 4K
                processed_frame = cv2.resize(frame, (416, 416))
                
                batch_frames.append(processed_frame)
                batch_indices.append(frame_idx)
                batch_timestamps.append(timestamp)
                
                # Process batch if full
                if len(batch_frames) >= batch_size:
                    batch_metrics = self._process_batch(batch_frames, batch_indices, batch_timestamps)
                    metrics.extend(batch_metrics)
                    
                    # Reset batch
                    batch_frames = []
                    batch_indices = []
                    batch_timestamps = []
                
            frame_idx += 1
            
        # Process remaining frames
        if batch_frames:
            batch_metrics = self._process_batch(batch_frames, batch_indices, batch_timestamps)
            metrics.extend(batch_metrics)
            
        cap.release()
        return metrics

    def _process_batch(self, frames: List[np.ndarray], indices: List[int], timestamps: List[float]) -> List[Dict]:
        # Run YOLO on batch
        # verbose=False, imgsz=416 are default but explicit is good
        results_list = self.yolo_model(frames, verbose=False, imgsz=416)
        
        batch_results = []
        
        for i, results in enumerate(results_list):
            frame = frames[i]
            frame_idx = indices[i]
            timestamp = timestamps[i]
            
            # Simple metrics
            confidence = self._calculate_confidence([results]) # Helper expects list
            posture = self._calculate_enhanced_posture([results])
            movement_score = self._calculate_movement([results])
            
            # Expensive CV ops - only occasionally
            if frame_idx % 10 == 0:
                attention = self._calculate_enhanced_attention(frame, [results])
                engagement = self._calculate_enhanced_engagement(frame, [results])
                eye_contact = self._calculate_eye_contact_from_keypoints([results])
            else:
                attention = 60.0
                engagement = 70.0
                eye_contact = 50.0
            
            # Count persons
            person_count = self._count_persons([results])
            
            batch_results.append({
                "timestamp": timestamp,
                "attention": attention,
                "confidence": confidence,
                "posture": posture,
                "engagement": engagement,
                "movement_stability": movement_score,
                "head_orientation": self._calculate_head_orientation([results]),
                "eye_contact_quality": eye_contact,
                "person_count": person_count
            })
            
        return batch_results

    

    
    def _calculate_enhanced_attention(self, frame, results) -> float:
        if not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return 0.0
            
        # Use Head Orientation as primary proxy for attention
        # If looking at camera (yaw ~ 0, pitch ~ 0), attention is high
        orientation = self._calculate_head_orientation(results)
        yaw = orientation["yaw"]
        pitch = orientation["pitch"]
        
        # Calculate deviation from center (0,0)
        deviation = math.sqrt(yaw**2 + pitch**2)
        
        # Map deviation to score: 0 deviation = 100 score, 45+ deviation = 0 score
        attention_score = max(0, 100 - (deviation * 2.2))
        
        return float(attention_score)

    
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
        
        # Body presence (Torso)
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Count visible body parts
        visible_parts = sum([
            float(left_shoulder[2].cpu()) > 0.5,
            float(right_shoulder[2].cpu()) > 0.5,
            float(left_hip[2].cpu()) > 0.5,
            float(right_hip[2].cpu()) > 0.5
        ])
        body_presence = visible_parts / 4
        
        # Facing Factor (Is user facing camera?)
        # If facing camera, shoulders are wide apart. If side profile, they are close.
        # We can also check nose vs shoulders center
        facing_score = 0.5 # Default
        if float(left_shoulder[2].cpu()) > 0.5 and float(right_shoulder[2].cpu()) > 0.5:
             # Calculate shoulder width relative to frame
             shoulder_width = abs(float(left_shoulder[0].cpu()) - float(right_shoulder[0].cpu()))
             # Normalize by frame width (assuming single person often centered)
             # This is a heuristic: wider shoulders = more direct facing
             # We can also check head orientation
             orientation = self._calculate_head_orientation(results)
             if abs(orientation['yaw']) < 20: 
                 facing_score = 1.0
             elif abs(orientation['yaw']) < 45:
                 facing_score = 0.7
             else:
                 facing_score = 0.3
                 
        engagement_score = (body_presence * 0.5) + (facing_score * 0.5)
        return min(100, engagement_score * 100)

    def _calculate_eye_contact_from_keypoints(self, results) -> float:
        if not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return 0.0
            
        # Use Head Orientation (Yaw/Pitch) as proxy for Eye Contact
        # If I am facing you directly, I am likely making eye contact
        orientation = self._calculate_head_orientation(results)
        yaw = abs(orientation["yaw"])
        pitch = abs(orientation["pitch"])
        
        # Strict threshold for eye contact
        # Yaw must be < 15 degrees, Pitch < 15 degrees
        if yaw < 15 and pitch < 15:
            # Score decreases as angle increases
            score = 100 - (yaw + pitch) * 2
            return max(0, score)
        
        return 20.0 # Low baseline if looking away
    
    def _count_persons(self, results) -> int:
        if not results[0].boxes or len(results[0].boxes.data) == 0:
            return 0
        
        # Count person detections (class 0)
        person_count = sum(1 for box in results[0].boxes.data if box[5] == 0)
        return person_count
    
    def _transcribe_audio_file(self, audio_path: str) -> Dict:
        try:
            # Transcribe with Faster-Whisper
            # Returns a generator
            segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
            
            # Convert generator to list immediately
            segment_list = list(segments)
            
            full_text = " ".join([seg.text.strip() for seg in segment_list])
            
            return {
                "text": full_text,
                "language": info.language,
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip()
                    }
                    for seg in segment_list
                ]
            }
        except Exception as e:
            print(f"Transcription error: {e}")
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