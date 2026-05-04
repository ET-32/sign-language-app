import csv
import os
import shutil
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'signs.csv')


def _normalize_hand(coords_63):
    pts = np.array(coords_63).reshape(21, 3)
    if np.allclose(pts, 0):
        return coords_63          # zero-padding hand — leave untouched
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().tolist()


def main():
    if not os.path.exists(DATA_PATH):
        print(f'ERROR: {DATA_PATH} not found.')
        return

    backup = DATA_PATH.replace('.csv', '_raw_backup.csv')
    shutil.copy(DATA_PATH, backup)
    print(f'Backup saved: {backup}')

    rows = []
    with open(DATA_PATH, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            label = row[0]
            vals  = [float(v) for v in row[1:]]
            h1    = _normalize_hand(vals[:63])
            h2    = _normalize_hand(vals[63:])
            rows.append([label] + h1 + h2)

    with open(DATA_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f'Re-normalized {len(rows)} samples across {len(set(r[0] for r in rows))} signs.')
    print('Run py -m src.train_model to retrain.')


if __name__ == '__main__':
    main()
