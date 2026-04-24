"""
Flask web server — serves the web UI, handles frame prediction, and WebRTC signaling.
"""

import base64
import os
import random
import string
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room

_ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')
_pred    = None
_lock    = threading.Lock()

# room_id -> set of session IDs
_rooms      = {}
_rooms_lock = threading.Lock()

import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)


def _get_predictor():
    global _pred
    if _pred is None:
        from src.predictor import SignPredictor
        _pred = SignPredictor()
    return _pred


def _make_room_id():
    """Generate a unique 6-char room code and reserve it."""
    with _rooms_lock:
        while True:
            rid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            if rid not in _rooms:
                _rooms[rid] = set()
                return rid


# ── HTTP Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(_ASSETS, 'index.html')



@app.route('/create-room')
def create_room():
    rid = _make_room_id()
    return jsonify({'roomId': rid})


@app.route('/sign_poses')
def sign_poses():
    return send_from_directory(_ASSETS, 'sign_poses.json', mimetype='application/json')


@app.route('/signs/<letter>')
def serve_sign(letter):
    """Serve locally stored ASL fingerspelling GIFs."""
    clean = ''.join(c for c in letter.lower() if c.isalpha())
    if len(clean) != 1:
        return '', 404
    signs_dir = os.path.join(_ASSETS, 'signs')
    fname = clean + '.gif'
    if os.path.exists(os.path.join(signs_dir, fname)):
        return send_from_directory(signs_dir, fname, mimetype='image/gif',
                                   max_age=86400)
    return '', 404


@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json(force=True, silent=True) or {}
    img_b64 = data.get('image', '')

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

    _, buf  = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    ann_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

    return jsonify({
        'label':      label,
        'confidence': round(float(conf), 3) if conf else 0,
        'image':      ann_b64,
    })


# ── Socket.IO — WebRTC Signaling ───────────────────────────────────────────────

@socketio.on('join-room')
def on_join(data):
    rid = (data.get('roomId') or '').strip().upper()
    if not rid:
        emit('call-error', {'msg': 'Invalid room code.'})
        return

    with _rooms_lock:
        if rid not in _rooms:
            emit('call-error', {'msg': 'Room not found. Check the code.'})
            return
        if len(_rooms[rid]) >= 2:
            emit('call-error', {'msg': 'Room is full (max 2 people).'})
            return
        _rooms[rid].add(request.sid)
        count         = len(_rooms[rid])
        initiator_sid = list(_rooms[rid] - {request.sid})[0] if count == 2 else None

    join_room(rid)
    emit('room-joined', {'roomId': rid, 'count': count})

    if initiator_sid:
        # Tell the first person (who is waiting) to create the WebRTC offer
        emit('peer-ready', {}, to=initiator_sid)


@socketio.on('call-speech')
def on_call_speech(data):
    rid = data.get('roomId', '')
    with _rooms_lock:
        others = list(_rooms.get(rid, set()) - {request.sid})
    for sid in others:
        emit('peer-speech', data, to=sid)


@socketio.on('video-frame')
def on_video_frame(data):
    rid = data.get('roomId', '')
    with _rooms_lock:
        others = list(_rooms.get(rid, set()) - {request.sid})
    for sid in others:
        emit('peer-video-frame', {'frame': data.get('frame')}, to=sid)


@socketio.on('sign-update')
def on_sign_update(data):
    rid = data.get('roomId', '')
    with _rooms_lock:
        others = list(_rooms.get(rid, set()) - {request.sid})
    for sid in others:
        emit('peer-sign', data, to=sid)


@socketio.on('disconnect')
def on_disconnect():
    others = []
    with _rooms_lock:
        for rid in list(_rooms.keys()):
            if request.sid in _rooms[rid]:
                _rooms[rid].discard(request.sid)
                others = list(_rooms[rid])
                if not _rooms[rid]:
                    del _rooms[rid]
                break
    for sid in others:
        emit('peer-left', {}, to=sid)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(host='0.0.0.0', port=5055):
    socketio.run(app, host=host, port=port, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
