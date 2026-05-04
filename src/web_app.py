import base64
import os
import random
import string
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_login import LoginManager, login_user, logout_user, current_user
from flask_socketio import SocketIO, emit, join_room

from src.models import db, User

_ASSETS  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'signai.db'))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'signai-dev-key-change-in-prod')

_db_url = os.environ.get('DATABASE_URL', f'sqlite:///{_DB_PATH}')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI']        = _db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager(app)
socketio = SocketIO(app, cors_allowed_origins='*')

_pred       = None
_lock       = threading.Lock()
_rooms      = {}
_rooms_lock = threading.Lock()

import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def _get_predictor():
    global _pred
    if _pred is None:
        from src.predictor import SignPredictor
        _pred = SignPredictor()
    return _pred


def _make_room_id():
    with _rooms_lock:
        while True:
            rid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            if rid not in _rooms:
                _rooms[rid] = set()
                return rid


def _user_dict(user):
    return {
        'username':        user.username,
        'is_subscribed':   user.is_subscribed,
        'calls_remaining': user.calls_remaining(),
        'can_call':        user.can_call(),
    }


@app.route('/api/register', methods=['POST'])
def api_register():
    data     = request.get_json() or {}
    username = (data.get('username') or '').strip()
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not username or not email or not password:
        return jsonify({'error': 'All fields are required.'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters.'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters.'}), 400
    if '@' not in email:
        return jsonify({'error': 'Enter a valid email address.'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken.'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered.'}), 409

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user, remember=True)
    return jsonify({'ok': True, 'user': _user_dict(user)})


@app.route('/api/login', methods=['POST'])
def api_login():
    data     = request.get_json() or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'Wrong username or password.'}), 401

    login_user(user, remember=True)
    return jsonify({'ok': True, 'user': _user_dict(user)})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    logout_user()
    return jsonify({'ok': True})


@app.route('/api/me')
def api_me():
    if not current_user.is_authenticated:
        return jsonify({'error': 'not_logged_in'}), 401
    return jsonify(_user_dict(current_user))


@app.route('/')
def index():
    return send_from_directory(_ASSETS, 'index.html')


@app.route('/create-room')
def create_room():
    if not current_user.is_authenticated:
        return jsonify({'error': 'login_required'}), 401
    if not current_user.can_call():
        return jsonify({
            'error': 'limit_reached',
            'msg':   'Monthly call limit reached. Upgrade to Pro for unlimited calls.',
        }), 403
    current_user.record_call()
    db.session.commit()
    rid = _make_room_id()
    return jsonify({'roomId': rid})


@app.route('/signs/<path:name>')
def serve_sign(name):
    clean = ''.join(c for c in name.lower() if c.isalpha())
    if not clean or len(clean) > 20:
        return '', 404
    signs_dir = os.path.join(_ASSETS, 'signs')
    fname = clean + '.gif'
    if os.path.exists(os.path.join(signs_dir, fname)):
        return send_from_directory(signs_dir, fname, mimetype='image/gif', max_age=86400)
    return '', 404


@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json(force=True, silent=True) or {}
    img_b64 = data.get('image', '')
    annotate = data.get('annotate', True)

    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]

    try:
        img_bytes = base64.b64decode(img_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'label': None, 'confidence': 0})

    if frame is None:
        return jsonify({'label': None, 'confidence': 0})

    try:
        predictor = _get_predictor()
    except FileNotFoundError:
        return jsonify({
            'label': None, 'confidence': 0,
            'error': 'Model not trained yet — run: py -m src.train_model'
        })

    with _lock:
        annotated, label, conf = predictor.process_frame(frame, draw=annotate)

    result = {
        'label':      label,
        'confidence': round(float(conf), 3) if conf else 0,
    }

    if annotate:
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        result['image'] = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

    return jsonify(result)


@socketio.on('join-room')
def on_join(data):
    if not current_user.is_authenticated:
        emit('call-error', {'msg': 'You must be logged in to join a call.'})
        return
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


@socketio.on('sign-trigger')
def on_sign_trigger(data):
    rid = data.get('roomId', '')
    with _rooms_lock:
        others = list(_rooms.get(rid, set()) - {request.sid})
    for sid in others:
        emit('peer-sign-trigger', data, to=sid)


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


def run(host='0.0.0.0', port=5055):
    socketio.run(app, host=host, port=port, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
