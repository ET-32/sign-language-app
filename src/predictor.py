"""
Real-Time Sign Predictor — supports up to 2 hands (126 landmarks).
"""

import os
import pickle
import numpy as np
import mediapipe as mp

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sign_model.pkl')

mp_hands = mp.solutions.hands


def _normalize_hand(coords_63):
    """
    Normalize 21-landmark hand coords to be position- and scale-invariant.
    Centers on wrist (landmark 0), scales by wrist-to-middle-MCP distance (landmark 9).
    This makes predictions the same regardless of where/how far the hand is from the camera.
    """
    pts = np.array(coords_63).reshape(21, 3)
    pts -= pts[0]                        # translate: wrist becomes origin
    scale = np.linalg.norm(pts[9])       # palm size = wrist → middle finger base
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().tolist()


def extract_landmarks(result):
    """
    Extract normalized landmarks from a MediaPipe result into a fixed 126-feature vector.
    Two hands × 21 landmarks × 3 (x, y, z) = 126.
    Each hand is normalized: wrist-centered and palm-scaled so predictions are
    invariant to hand position in frame and distance from camera.
    Hands are ordered left-to-right by wrist x. Pads with zeros for missing hands.
    """
    hand_data = []
    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            wrist_x = hand_lm.landmark[0].x
            coords = []
            for lm in hand_lm.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            hand_data.append((wrist_x, coords))

        hand_data.sort(key=lambda h: h[0])  # left-to-right order

    features = []
    for _, coords in hand_data:
        features.extend(_normalize_hand(coords))

    features.extend([0.0] * (126 - len(features)))
    return features[:126]


class SignPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: py -m src.train_model"
            )
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Takes a BGR frame, returns (annotated_frame, label, confidence).
        """
        import cv2
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        prediction = None
        confidence = None

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS
                )

            features = extract_landmarks(result)
            X        = np.array(features).reshape(1, -1)
            prediction = self.model.predict(X)[0]
            proba      = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))

        return frame, prediction, confidence

    def close(self):
        self.hands.close()
