import cv2
import time
import pandas as pd
from deepface import DeepFace
from advanced_layer import AdvancedAnalyzer


class EmotionEngine:
    """
    Core emotion analysis engine.
    ONLY handles emotion logic.
    NO Streamlit code here.
    """

    def __init__(self, frame_interval=1):
        self.frame_interval = frame_interval
        self.adv = AdvancedAnalyzer()

    # ----------------------------
    # Analyze a single frame
    # ----------------------------
    def _analyze_frame(self, frame, current_time):
        # ---- DeepFace emotion detection ----
        try:
            analysis = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )
            emotion_scores = analysis[0]["emotion"]
            detected_emotion = analysis[0]["dominant_emotion"]
            confidence = emotion_scores.get(detected_emotion, 50.0)

        except Exception:
            detected_emotion = "neutral"
            confidence = 40.0

        # ---- Advanced facial features ----
        features = self.adv.analyze_frame(frame)

        if features is None:
            return {
                "Time (s)": current_time,
                "Emotion": detected_emotion,
                "Confidence": confidence,
                "Blink": 0,
                "Eye_Distance": None,
                "Mouth_Open": None,
                "Mouth_Width": None,
                "Eyebrow_Lift": None,
                "Nose_X": None,
                "Nose_Flare": None
            }

        mouth_open = features["mouth_open"]
        mouth_width = features["mouth_width"]
        eyebrow_lift = features["eyebrow_lift"]

        # ----------------------------
        # Emotion correction rules
        # ----------------------------

        # ANGRY: tight eyebrows + closed mouth
        if eyebrow_lift is not None and mouth_open is not None:
            if eyebrow_lift < -0.03 and mouth_open < 0.02:
                detected_emotion = "angry"
                confidence = max(confidence, 80.0)

        # HAPPY: relaxed smile
        if mouth_open is not None and mouth_width is not None:
            if mouth_open > 0.028 and mouth_width > 0.11:
                detected_emotion = "happy"
                confidence = max(confidence, 80.0)

        # FEAR: wide eyes + raised eyebrows + no smile
        if (
            eyebrow_lift is not None
            and mouth_open is not None
            and mouth_width is not None
            and eyebrow_lift > 0.02
            and 0.02 < mouth_open < 0.045
            and mouth_width < 0.105
        ):
            detected_emotion = "fear"
            confidence = max(confidence, 82.0)

        # Suppress weak fear
        if detected_emotion == "fear" and confidence < 85:
            detected_emotion = "neutral"

        return {
            "Time (s)": current_time,
            "Emotion": detected_emotion,
            "Confidence": confidence,
            "Blink": int(features["blink"]),
            "Eye_Distance": features["avg_eye_dist"],
            "Mouth_Open": mouth_open,
            "Mouth_Width": mouth_width,
            "Eyebrow_Lift": eyebrow_lift,
            "Nose_X": features["nose_x"],
            "Nose_Flare": features["nose_flare"]
        }

    # ----------------------------
    # Webcam analysis (FINITE)
    # ----------------------------
    def analyze_webcam(self, duration=10):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        results = []

        while int(time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = int(time.time() - start_time)

            if current_time % self.frame_interval == 0:
                row = self._analyze_frame(frame, current_time)
                results.append(row)

        cap.release()
        return pd.DataFrame(results)

    # ----------------------------
    # Video file analysis
    # ----------------------------
    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        frame_count = 0
        results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = int(frame_count / fps)

            if frame_count % int(fps * self.frame_interval) == 0:
                row = self._analyze_frame(frame, current_time)
                results.append(row)

        cap.release()
        return pd.DataFrame(results)
    
    def analyze_frames(self, frames):
        results = []
        start_time = time.time()
    
        for i, frame in enumerate(frames):
            current_time = int(time.time() - start_time)
    
            if i % self.frame_interval == 0:
                row = self._analyze_frame(frame, current_time)
                results.append(row)
    
        return pd.DataFrame(results)


    # ----------------------------
    # Save results
    # ----------------------------
    @staticmethod
    def save_to_csv(df, filename="emotion_results.csv"):
        df.to_csv(filename, index=False)
