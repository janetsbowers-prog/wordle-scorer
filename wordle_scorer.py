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
            
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect grid cells based on color regions
        rows = []
        
        # Define color ranges in HSV for detection
        # Green tiles
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        
        # Yellow tiles  
        yellow_lower = np.array([15, 100, 100])
        yellow_upper = np.array([35, 255, 255])
        
        # Gray tiles (wrong letters)
        gray_lower = np.array([0, 0, 40])
        gray_upper = np.array([180, 50, 140])
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create masks for each color
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        
        # Find contours to identify individual cells
        combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours by position
        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (cells should be roughly square and similar size)
            if w > 20 and h > 20 and 0.7 < w/h < 1.3:
                # Determine color
                cell_roi = hsv[y:y+h, x:x+w]
                
                green_pixels = cv2.inRange(cell_roi, green_lower, green_upper).sum()
                yellow_pixels = cv2.inRange(cell_roi, yellow_lower, yellow_upper).sum()
                
                if green_pixels > yellow_pixels:
                    color = 'green'
                elif yellow_pixels > 0:
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
                        'letter': letter,
                        'color': color
                    })
        
        # Sort cells into rows (top to bottom, left to right)
        cells.sort(key=lambda c: (c['y'], c['x']))
        
        # Group into rows of 5
        for i in range(0, len(cells), 5):
            row = cells[i:i+5]
            if len(row) == 5:
                rows.append(row)
                
        return rows
    
    def _extract_letter(self, cell_img: np.ndarray) -> Optional[str]:
        """Extract letter from cell image using OCR"""
        # Preprocess for better OCR
        gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
        
        # Threshold to get white letters
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Use pytesseract with config for single uppercase letter
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        text = pytesseract.image_to_string(thresh, config=config).strip()
        
        # Return first letter if valid
        if text and len(text) >= 1 and text[0].isalpha():
            return text[0].upper()
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
                    'error': 'Could not detect Wordle grid in image'
                }
                
            result = self.calculate_score(rows)
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
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
