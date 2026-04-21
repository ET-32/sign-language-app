"""
Entry point — starts the Flask web server and opens the browser.
Run with: py main.py
"""

import sys
import os
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(__file__))

from src.web_app import run

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5055))

    # On cloud (PORT env var set) don't try to open a browser
    if not os.environ.get('PORT'):
        threading.Timer(1.2, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()

    run(host='0.0.0.0', port=port)
