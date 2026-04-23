"""
Data Collection — records hand landmark data for each sign.

Menu-driven: pick a sign by number, hold the pose shown on screen,
press SPACE to start auto-capture (~20 seconds per sign). Press Q to quit.

Two hands are supported — for signs like HELP and STOP, show both hands.
"""

import csv
import os
import time

import cv2
import mediapipe as mp

from src.predictor import extract_landmarks

DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'data', 'signs.csv')
SAMPLES_PER_SIGN = 100
CAPTURE_INTERVAL = 0.18   # seconds between auto-captures

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# Signs to collect and the exact pose to hold during recording
ALL_SIGNS = [
    'HELLO',
    'GOOD',
    'THANK YOU',
    'YES',
    'NO',
    'OK',
    'I LOVE YOU',
    'CALL ME',
    'PEACE',
    'SORRY',
    'PLEASE',
    'HELP',
    'STOP',
]

SIGN_GUIDE = {
    'HELLO':     'Open flat hand near forehead, fingers together (like a salute)',
    'GOOD':      'Flat hand starts at chin, hold it extended forward',
    'THANK YOU': 'Flat hand at lips/chin, hold it extended outward',
    'YES':       'Closed fist (S-hand) — hold still, no need to nod',
    'NO':        'Index + middle fingers bent down touching thumb',
    'OK':        'Index fingertip touches thumb tip, other 3 fingers up',
    'I LOVE YOU':'Thumb + index finger + pinky extended (ILY handshape)',
    'CALL ME':   'Pinky + thumb extended, other fingers curled (Y-shape)',
    'PEACE':     'Index + middle fingers raised in V shape',
    'SORRY':     'Closed fist held flat against chest',
    'PLEASE':    'Flat open palm pressed against chest',
    'HELP':      'TWO HANDS — thumbs-up right hand resting on flat left palm',
    'STOP':      'TWO HANDS — right open hand pressed perpendicular into left palm',
}

EXPECTED_FEATURES = 126   # 2 hands × 21 landmarks × 3
EXPECTED_COLS     = EXPECTED_FEATURES + 1  # +1 for label column


def check_csv_format():
    """Return True if the existing CSV matches the 2-hand format, False if old format."""
    if not os.path.exists(DATA_PATH):
        return True  # no file yet, fine
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header is None:
        return True
    return len(header) == EXPECTED_COLS


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


def collect_sign(cap, hands, writer, sign, existing_count):
    needed    = SAMPLES_PER_SIGN - existing_count
    count     = existing_count
    if needed <= 0:
        return count

    capturing    = False
    last_capture = 0.0
    guide        = SIGN_GUIDE.get(sign, '')
    two_handed   = sign in ('HELP', 'STOP')
    status_msg   = 'Press SPACE to start auto-capture'

    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        num_hands = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        hand_ok   = num_hands >= 2 if two_handed else num_hands >= 1

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # Auto-capture
        now = time.time()
        if capturing and hand_ok and (now - last_capture) >= CAPTURE_INTERVAL:
            features     = extract_landmarks(result)
            writer.writerow([sign] + features)
            count        += 1
            last_capture  = now

        # HUD
        h, w = frame.shape[:2]
        collected = count - existing_count
        pct = int(collected / needed * w) if needed else w
        cv2.rectangle(frame, (0, h - 8), (pct, h), (0, 220, 100), -1)

        hdr_col = (0, 220, 100) if capturing else (0, 180, 255)
        cv2.putText(frame, f'Sign: {sign}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, hdr_col, 2)
        cv2.putText(frame, f'Samples: {count}/{SAMPLES_PER_SIGN}', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Pose guide
        for gi, line in enumerate(guide.split(' — ')):
            cv2.putText(frame, line, (10, 85 + gi * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1)

        # Hand detection status
        if two_handed:
            hand_txt = f'{num_hands}/2 hands detected'
            hand_col = (0, 255, 0) if hand_ok else (0, 120, 255)
        else:
            hand_txt = 'Hand detected' if hand_ok else 'No hand detected'
            hand_col = (0, 255, 0)    if hand_ok else (0, 0, 255)
        cv2.putText(frame, hand_txt, (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_col, 1)

        state_txt = 'AUTO-CAPTURING...' if capturing else status_msg
        state_col = (0, 220, 100)       if capturing else (200, 200, 80)
        cv2.putText(frame, state_txt, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_col, 1)
        cv2.putText(frame, 'SPACE: start/stop  |  B: back  |  Q: quit',
                    (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if hand_ok:
                capturing    = not capturing
                last_capture  = time.time()
                status_msg   = 'Paused — press SPACE to resume' if not capturing else ''
            else:
                need_txt = 'Show BOTH hands first' if two_handed else 'Show your hand first'
                status_msg = need_txt

        elif key in (ord('b'), ord('B')):
            break
        elif key in (ord('q'), ord('Q')):
            count = -1
            break

    return count


def main():
    # Check for incompatible old format
    if not check_csv_format():
        print('\nWARNING: Existing signs.csv uses the OLD format (1 hand, 63 features).')
        print('The new format supports 2 hands (126 features) — the files are incompatible.')
        ans = input('Back up old data and start fresh? (yes/no): ').strip().lower()
        if ans == 'yes':
            backup = DATA_PATH.replace('.csv', '_backup_1hand.csv')
            os.rename(DATA_PATH, backup)
            print(f'Old data backed up to: {backup}')
        else:
            print('Keeping old file. You will need to delete it manually to use the new format.')
            return

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    file_exists = os.path.exists(DATA_PATH)
    csvfile = open(DATA_PATH, 'a', newline='')
    writer  = csv.writer(csvfile)
    if not file_exists:
        header = ['label'] + [f'{ax}{i}' for i in range(42) for ax in ('x', 'y', 'z')]
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
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            counts     = load_existing_counts()
            total_done = sum(1 for s in ALL_SIGNS if counts.get(s, 0) >= SAMPLES_PER_SIGN)

            print(f'\n{"="*50}')
            print(f'Progress: {total_done}/{len(ALL_SIGNS)} signs complete')
            print(f'{"="*50}')
            for i, s in enumerate(ALL_SIGNS):
                c      = counts.get(s, 0)
                bar    = '#' * int(c / SAMPLES_PER_SIGN * 20)
                status = 'DONE' if c >= SAMPLES_PER_SIGN else f'{c}/{SAMPLES_PER_SIGN}'
                two    = ' (2 hands)' if s in ('HELP', 'STOP') else ''
                print(f'  {i+1:2}. {s:<14}{two:<12} [{bar:<20}] {status}')

            print('\nEnter number to collect, or Q to quit and train:')
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

            sign     = ALL_SIGNS[idx]
            existing = counts.get(sign, 0)

            print(f'\nCollecting: {sign}')
            print(f'Pose: {SIGN_GUIDE.get(sign, "")}')
            if sign in ('HELP', 'STOP'):
                print('NOTE: Show BOTH hands in the frame.')
            print('Press SPACE to start auto-capture. B to go back.')

            result = collect_sign(cap, hands, writer, sign, existing)
            csvfile.flush()

            if result == -1:
                break
            if result >= SAMPLES_PER_SIGN:
                print(f'"{sign}" complete! {result} samples saved.')

    cap.release()
    cv2.destroyAllWindows()
    csvfile.close()

    counts     = load_existing_counts()
    total_done = sum(1 for s in ALL_SIGNS if counts.get(s, 0) >= SAMPLES_PER_SIGN)
    print(f'\nDone. {total_done}/{len(ALL_SIGNS)} signs ready.')
    if total_done > 0:
        print('\nTo train the model run:')
        print('  py -m src.train_model')


if __name__ == '__main__':
    main()
