import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh

class AdvancedAnalyzer:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.eye_closed_prev = False
        self.blink_count = 0

    def analyze_frame(self, frame):
        """
        Analyze one frame and return facial metrics.
        Returns None if no face is detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Eye landmarks
        left_upper = landmarks[159].y
        left_lower = landmarks[145].y
        right_upper = landmarks[386].y
        right_lower = landmarks[374].y

        left_eye_dist = left_lower - left_upper
        right_eye_dist = right_lower - right_upper
        avg_eye_dist = (left_eye_dist + right_eye_dist) / 2

        # Blink detection
        BLINK_THRESHOLD = 0.02
        blink = False
        if avg_eye_dist < BLINK_THRESHOLD:
            if not self.eye_closed_prev:
                self.blink_count += 1
                blink = True
                self.eye_closed_prev = True
        else:
            self.eye_closed_prev = False

        # Mouth open
        mouth_open = abs(landmarks[13].y - landmarks[14].y)

        # Mouth width (smile detection)
        mouth_width = abs(landmarks[61].x - landmarks[291].x)

        # Eyebrow lift (forehead activity)
        eyebrow_lift = landmarks[70].y - landmarks[159].y  # smaller → more raised

        # Gaze direction (nose x)
        nose_x = landmarks[1].x

        # Nose flare (nostril expansion)
        nose_flare = abs(landmarks[98].x - landmarks[327].x)

        return {
            "blink": blink,
            "avg_eye_dist": float(avg_eye_dist),
            "mouth_open": float(mouth_open),
            "mouth_width": float(mouth_width),
            "eyebrow_lift": float(eyebrow_lift),
            "nose_x": float(nose_x),
            "nose_flare": float(nose_flare)
        }
