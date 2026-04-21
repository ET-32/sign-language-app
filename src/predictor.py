"""
Phase 5: Real-Time Sign Predictor
Loads the trained model and predicts signs from live webcam input.
Used by ui.py to get predictions each frame.
"""

import os
import pickle
import numpy as np
import mediapipe as mp

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sign_model.pkl')

mp_hands = mp.solutions.hands


class SignPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run src/train_model.py first."
            )
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Takes a BGR frame, runs hand detection, returns:
          - annotated_frame: frame with landmarks drawn
          - prediction: predicted sign label string, or None
          - confidence: float 0-1, or None
        """
        import cv2
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        prediction = None
        confidence = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            X = np.array(coords).reshape(1, -1)
            prediction = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))

        return frame, prediction, confidence

    def close(self):
        self.hands.close()
