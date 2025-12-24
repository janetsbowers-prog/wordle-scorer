#!/usr/bin/env python3
"""
Flask API server for Wordle Scorer
Provides REST endpoint for image processing
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
from io import BytesIO
from PIL import Image
from wordle_scorer import WordleScorer
import tempfile

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for development

scorer = WordleScorer()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'wordle_scorer.html')

@app.route('/api/score', methods=['POST'])
def score_wordle():
    """
    Process uploaded Wordle image and return score
    Expects: JSON with base64 encoded image
    Returns: JSON with score and letter breakdown
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Decode base64 image
        image_data = data['image']
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode and save to temporary file
        image_bytes = base64.b64decode(image_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Process the image
            result = scorer.process_image(tmp_path)
            return jsonify(result)
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Check dependencies
    try:
        import cv2
        import pytesseract
        print("✓ OpenCV and Tesseract found")
    except ImportError as e:
        print(f"⚠ Missing dependency: {e}")
        print("Install with: pip install opencv-python pytesseract pillow flask flask-cors")
        exit(1)
    
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Starting Wordle Scorer API...")
    print(f"📍 Server running on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
