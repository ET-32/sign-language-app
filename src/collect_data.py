"""
Phase 3: Data Collection
Records hand landmark data for each sign and saves it to data/signs.csv.

Controls:
  - Type the sign label when prompted (e.g. HELLO, A, B, YES)
  - Press SPACE to capture a sample
  - Press Q to quit and save
"""

import cv2
import mediapipe as mp
import csv
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'signs.csv')
SAMPLES_PER_SIGN = 100

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Write CSV header if file doesn't exist
    file_exists = os.path.exists(DATA_PATH)
    csvfile = open(DATA_PATH, 'a', newline='')
    writer = csv.writer(csvfile)
    if not file_exists:
        header = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
        writer.writerow(['label'] + header)

    label = input("Enter the sign label to record (e.g. HELLO, A, YES): ").strip().upper()
    print(f"Recording '{label}'. Press SPACE to capture a sample. Press Q to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            landmarks = None
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = result.multi_hand_landmarks[0]

            status = f"Label: {label} | Samples: {count}/{SAMPLES_PER_SIGN}"
            cv2.putText(frame, status, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if landmarks:
                cv2.putText(frame, "Hand detected - press SPACE to capture", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Phase 3 - Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and landmarks:
                row = [label] + extract_landmarks(landmarks)
                writer.writerow(row)
                count += 1
                print(f"  Captured sample {count}/{SAMPLES_PER_SIGN}")
                if count >= SAMPLES_PER_SIGN:
                    print(f"Done! {SAMPLES_PER_SIGN} samples collected for '{label}'.")
                    break

            elif key == ord('q'):
                print(f"Quit. {count} samples saved for '{label}'.")
                break

    cap.release()
    cv2.destroyAllWindows()
    csvfile.close()
    print(f"Data saved to {DATA_PATH}")


if __name__ == '__main__':
    main()
