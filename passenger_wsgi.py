import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import app  # import your Flask app

# Enable CORS for the frontend domain
from flask_cors import CORS

CORS(app, resources={r"/predict": {"origins": [
    "https://garbageclassification.insaash.space",
    "http://127.0.0.1:5000"
]}})
# This is required by WSGI servers (Apache + mod_wsgi)
application = app
