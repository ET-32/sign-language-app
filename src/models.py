from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

FREE_CALLS_PER_MONTH = 10


class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_subscribed = db.Column(db.Boolean, default=False)
    call_count    = db.Column(db.Integer, default=0)
    call_month    = db.Column(db.Integer, default=0)
    call_year     = db.Column(db.Integer, default=0)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def _sync_month(self):
        now = datetime.utcnow()
        if self.call_month != now.month or self.call_year != now.year:
            self.call_count = 0
            self.call_month = now.month
            self.call_year  = now.year

    def calls_remaining(self):
        if self.is_subscribed:
            return None  # unlimited
        self._sync_month()
        return max(0, FREE_CALLS_PER_MONTH - self.call_count)

    def can_call(self):
        if self.is_subscribed:
            return True
        self._sync_month()
        return self.call_count < FREE_CALLS_PER_MONTH

    def record_call(self):
        self._sync_month()
        self.call_count += 1
