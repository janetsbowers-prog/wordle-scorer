#!/usr/bin/env python3
"""
Wordle Score Calculator - OCR Backend
Processes Wordle game board images and calculates scores
"""

from PIL import Image
import pytesseract
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import re

# Scoring configuration
SCORING = {
    1: {'green': 8, 'yellow': 4},
    2: {'green': 7, 'yellow': 3},
    3: {'green': 6, 'yellow': 2},
    4: {'green': 5, 'yellow': 1},
    5: {'green': 4, 'yellow': 0},
    6: {'green': 4, 'yellow': 0}
}
CONVERSION_BONUS = 1


class WordleScorer:
    """Analyzes Wordle game board images and calculates scores"""
    
    def __init__(self):
        self.letter_history = {}  # Track first appearance and conversions
        
    def detect_cells(self, image_path: str) -> List[List[Dict]]:
        """
        Detect the 5x6 grid of Wordle cells and extract letter/color info
        Returns: List of rows, each row is a list of cell dicts with 'letter' and 'color'
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        # Resize if image is too large (helps with processing)
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
            
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # More lenient color ranges for Wordle tiles
        # Green tiles - wider range to catch different shades
        green_lower = np.array([30, 30, 30])
        green_upper = np.array([90, 255, 255])
        
        # Yellow/Gold tiles - wider range
        yellow_lower = np.array([10, 80, 80])
        yellow_upper = np.array([40, 255, 255])
        
        # Gray tiles - much wider range
        gray_lower = np.array([0, 0, 30])
        gray_upper = np.array([180, 80, 200])
        
        # Create masks for each color
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        cells = []
        min_area = 500  # Minimum cell area in pixels
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter: reasonable size and roughly square
            if area > min_area and 0.6 < aspect_ratio < 1.4:
                # Determine color by checking which mask has most pixels
                cell_roi_hsv = hsv[y:y+h, x:x+w]
                
                green_pixels = cv2.inRange(cell_roi_hsv, green_lower, green_upper).sum()
                yellow_pixels = cv2.inRange(cell_roi_hsv, yellow_lower, yellow_upper).sum()
                gray_pixels = cv2.inRange(cell_roi_hsv, gray_lower, gray_upper).sum()
                
                # Determine dominant color
                if green_pixels > yellow_pixels and green_pixels > gray_pixels:
                    color = 'green'
                elif yellow_pixels > gray_pixels:
                    color = 'yellow'
                else:
                    color = 'gray'
                
                # Extract letter using OCR
                cell_img = img_rgb[y:y+h, x:x+w]
                letter = self._extract_letter(cell_img)
                
                if letter:
                    cells.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'letter': letter,
                        'color': color
                    })
        
        if len(cells) < 10:  # Need at least 2 rows worth
            return []
        
        # Sort cells by position (top to bottom, left to right)
        cells.sort(key=lambda c: (c['y'], c['x']))
        
        # Group into rows - use Y-coordinate clustering
        rows = []
        current_row = []
        last_y = cells[0]['y'] if cells else 0
        y_threshold = 20  # Pixels tolerance for same row
        
        for cell in cells:
            if abs(cell['y'] - last_y) > y_threshold and current_row:
                # New row detected
                if len(current_row) == 5:  # Valid Wordle row
                    # Sort current row by x coordinate
                    current_row.sort(key=lambda c: c['x'])
                    rows.append(current_row)
                current_row = [cell]
                last_y = cell['y']
            else:
                current_row.append(cell)
                
        # Don't forget the last row
        if len(current_row) == 5:
            current_row.sort(key=lambda c: c['x'])
            rows.append(current_row)
                
        return rows
    
    def _extract_letter(self, cell_img: np.ndarray) -> Optional[str]:
        """Extract letter from cell image using OCR"""
        # Preprocess for better OCR
        gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
        
        # Try multiple thresholding techniques
        # Method 1: Simple threshold
        _, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: Otsu's threshold
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try OCR on all three versions
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        for thresh in [thresh1, thresh2, thresh3]:
            try:
                text = pytesseract.image_to_string(thresh, config=config).strip()
                if text and len(text) >= 1 and text[0].isalpha():
                    return text[0].upper()
            except:
                continue
                
        return None
    
    def calculate_score(self, rows: List[List[Dict]]) -> Dict:
        """
        Calculate score based on letter appearances and conversions
        Returns dict with total score and letter breakdown
        """
        self.letter_history = {}
        letter_scores = []
        
        for row_idx, row in enumerate(rows, 1):
            for cell in row:
                letter = cell['letter']
                color = cell['color']
                
                # Skip gray letters (not in the word)
                if color == 'gray':
                    continue
                    
                # Track letter's first appearance
                if letter not in self.letter_history:
                    # First time seeing this letter
                    base_points = SCORING[row_idx][color]
                    
                    self.letter_history[letter] = {
                        'row': row_idx,
                        'first_color': color,
                        'base_points': base_points,
                        'converted': False
                    }
                    
                else:
                    # Letter appeared before - check for conversion
                    if self.letter_history[letter]['first_color'] == 'yellow' and color == 'green':
                        if not self.letter_history[letter]['converted']:
                            self.letter_history[letter]['converted'] = True
        
        # Calculate final scores for each letter
        total_score = 0
        for letter, info in self.letter_history.items():
            points = info['base_points']
            if info['converted']:
                points += CONVERSION_BONUS
                
            total_score += points
            
            letter_scores.append({
                'char': letter,
                'row': info['row'],
                'color': info['first_color'],
                'points': info['base_points'],
                'conversion': info['converted'],
                'totalPoints': points
            })
        
        # Sort by row for display
        letter_scores.sort(key=lambda x: x['row'])
        
        return {
            'success': True,
            'totalScore': total_score,
            'letters': letter_scores
        }
    
    def process_image(self, image_path: str) -> Dict:
        """Main processing function"""
        try:
            rows = self.detect_cells(image_path)
            
            if not rows:
                return {
                    'success': False,
                    'error': 'Could not detect Wordle grid in image. Please ensure the image clearly shows the Wordle game board.'
                }
                
            result = self.calculate_score(rows)
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }


def main():
    """Test function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wordle_scorer.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    scorer = WordleScorer()
    result = scorer.process_image(image_path)
    
    if result['success']:
        print(f"Total Score: {result['totalScore']}")
        print("\nLetter Breakdown:")
        for letter in result['letters']:
            conv_text = " + 1 conversion" if letter['conversion'] else ""
            print(f"  {letter['char']}: Row {letter['row']} {letter['color']}{conv_text} = {letter['totalPoints']} pts")
    else:
        print(f"Error: {result['error']}")


if __name__ == '__main__':
    main()
