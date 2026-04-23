"""
Data Collection — records hand landmark data for each sign.

Menu-driven: pick a sign, hold your hand in position, press SPACE
to start auto-capture (100 samples in ~20 seconds). Press Q to quit.
"""

import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'signs.csv')
SAMPLES_PER_SIGN = 100
CAPTURE_INTERVAL = 0.18   # seconds between auto-captures (~5-6 fps)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# All signs to collect — add/remove as needed
ALL_SIGNS = [
    'HELLO', 'GOOD', 'THANK YOU',
    'YES', 'NO', 'OK',
    'I LOVE YOU', 'CALL ME', 'PEACE',
    'SORRY', 'PLEASE', 'HELP',
    'STOP', 'WAIT', 'WHERE', 'WHAT',
]


def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def load_existing_counts():
    counts = {s: 0 for s in ALL_SIGNS}
    if not os.path.exists(DATA_PATH):
        return counts
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0] in counts:
                counts[row[0]] += 1
    return counts


def draw_menu(frame, counts, selected=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 14, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, 'SIGN LANGUAGE DATA COLLECTION', (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, 'Press the number to select a sign, Q to quit',
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    for i, sign in enumerate(ALL_SIGNS):
        cnt   = counts.get(sign, 0)
        done  = cnt >= SAMPLES_PER_SIGN
        color = (80, 200, 80) if done else (200, 200, 80) if cnt > 0 else (200, 200, 200)
        bar   = '#' * int(cnt / SAMPLES_PER_SIGN * 20)
        col   = 20 if i < 8 else 320
        row   = 90 + (i % 8) * 28
        label = f'{i+1:2}. {sign:<12} [{bar:<20}] {cnt}/{SAMPLES_PER_SIGN}'
        cv2.putText(frame, label, (col, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def collect_sign(cap, hands, writer, sign, existing_count):
    count   = existing_count
    needed  = SAMPLES_PER_SIGN - count
    if needed <= 0:
        return count

    capturing   = False
    last_capture = 0.0
    status_msg  = f'Hold the "{sign}" sign, then press SPACE to start auto-capture'

    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result   = hands.process(rgb)

        landmarks = None
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Auto-capture
        now = time.time()
        if capturing and landmarks and (now - last_capture) >= CAPTURE_INTERVAL:
            row_data = [sign] + extract_landmarks(landmarks)
            writer.writerow(row_data)
            count       += 1
            last_capture = now

        # HUD
        h, w = frame.shape[:2]
        pct  = int((count - existing_count) / needed * w)
        cv2.rectangle(frame, (0, h - 10), (pct, h), (0, 220, 100), -1)

        header_col = (0, 220, 100) if capturing else (0, 180, 255)
        cv2.putText(frame, f'Sign: {sign}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, header_col, 2)
        cv2.putText(frame, f'Samples: {count}/{SAMPLES_PER_SIGN}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        hand_txt = 'Hand detected' if landmarks else 'No hand detected'
        hand_col = (0, 255, 0)     if landmarks else (0, 0, 255)
        cv2.putText(frame, hand_txt, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_col, 1)

        state_txt = 'AUTO-CAPTURING...' if capturing else status_msg
        state_col = (0, 220, 100) if capturing else (200, 200, 80)
        cv2.putText(frame, state_txt, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_col, 1)
        cv2.putText(frame, 'SPACE: start/stop | B: back to menu | Q: quit',
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if landmarks:
                capturing    = not capturing
                last_capture = time.time()
                status_msg   = 'Paused — press SPACE to resume' if not capturing else ''
            else:
                status_msg = 'No hand detected — show your hand first'

        elif key in (ord('b'), ord('B')):
            break

        elif key in (ord('q'), ord('Q')):
            count = -1   # signal quit
            break

    return count


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    file_exists = os.path.exists(DATA_PATH)
    csvfile = open(DATA_PATH, 'a', newline='')
    writer  = csv.writer(csvfile)
    if not file_exists:
        header = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ('x', 'y', 'z')]
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Could not open webcam.')
        csvfile.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            counts = load_existing_counts()
            total_done = sum(1 for s in ALL_SIGNS if counts.get(s, 0) >= SAMPLES_PER_SIGN)
            print(f'\nProgress: {total_done}/{len(ALL_SIGNS)} signs complete.')
            print('Signs and current counts:')
            for i, s in enumerate(ALL_SIGNS):
                c = counts.get(s, 0)
                bar = '#' * int(c / SAMPLES_PER_SIGN * 20)
                status = 'DONE' if c >= SAMPLES_PER_SIGN else f'{c}/{SAMPLES_PER_SIGN}'
                print(f'  {i+1:2}. {s:<14} [{bar:<20}] {status}')

            print('\nEnter a number (1-16) to collect that sign, or Q to quit and train:')
            choice = input('> ').strip().upper()

            if choice in ('Q', 'QUIT', ''):
                break

            try:
                idx = int(choice) - 1
                if not (0 <= idx < len(ALL_SIGNS)):
                    print('Invalid number.')
                    continue
            except ValueError:
                print('Enter a number or Q.')
                continue

            sign          = ALL_SIGNS[idx]
            existing      = counts.get(sign, 0)
            if existing >= SAMPLES_PER_SIGN:
                print(f'"{sign}" already has {existing} samples. Collecting more will overwrite.')

            print(f'\nCollecting: {sign}')
            print('  - Show your hand in the webcam window')
            print('  - Press SPACE to start/stop auto-capture')
            print('  - Press B to go back to the menu')

            result = collect_sign(cap, hands, writer, sign, existing)
            csvfile.flush()

            if result == -1:   # quit signal
                break
            if result >= SAMPLES_PER_SIGN:
                print(f'"{sign}" complete! {result} samples saved.')

    cap.release()
    cv2.destroyAllWindows()
    csvfile.close()

    counts = load_existing_counts()
    total_done = sum(1 for s in ALL_SIGNS if counts.get(s, 0) >= SAMPLES_PER_SIGN)
    print(f'\nDone. {total_done}/{len(ALL_SIGNS)} signs have enough data.')
    if total_done > 0:
        print('\nRun this to train the model:')
        print('  py -m src.train_model')


if __name__ == '__main__':
    main()
