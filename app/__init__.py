from flask import Flask

# Create Flask app instance
app = Flask(__name__)

# Import routes after app creation to avoid circular imports
from app import routes
