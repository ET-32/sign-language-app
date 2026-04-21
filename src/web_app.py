"""
Flask web server — serves the web UI and handles frame prediction.
Browser sends camera frames → we run MediaPipe + model → return result.
"""

import base64
import os
import socket
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

_ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))

app      = Flask(__name__)
_pred    = None
_lock    = threading.Lock()

import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)


def _get_predictor():
    global _pred
    if _pred is None:
        from src.predictor import SignPredictor
        _pred = SignPredictor()
    return _pred


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(_ASSETS, 'index.html')


@app.route('/info')
def info():
    """Return the LAN URL so the browser can display a share link."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = '127.0.0.1'
    return jsonify({'local_url': f'http://{local_ip}:5055'})


@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json(force=True, silent=True) or {}
    img_b64 = data.get('image', '')

    # Strip data-URL prefix
    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]

    try:
        img_bytes = base64.b64decode(img_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'label': None, 'confidence': 0, 'image': None})

    if frame is None:
        return jsonify({'label': None, 'confidence': 0, 'image': None})

    try:
        predictor = _get_predictor()
    except FileNotFoundError:
        return jsonify({
            'label': None, 'confidence': 0, 'image': None,
            'error': 'Model not trained yet — run: py -m src.train_model'
        })

    with _lock:
        annotated, label, conf = predictor.process_frame(frame)

    _, buf    = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    ann_b64   = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

    return jsonify({
        'label':      label,
        'confidence': round(float(conf), 3) if conf else 0,
        'image':      ann_b64,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def run(host='0.0.0.0', port=5055):
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
