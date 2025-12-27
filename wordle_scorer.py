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

# Common OCR misreads for Wordle letters
OCR_CORRECTIONS = {
    '0': 'O',  # Zero to O
    '1': 'I',  # One to I
    '8': 'B',  # Eight to B
    '5': 'S',  # Five to S
    '6': 'G',  # Six to G
}


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
        
        # Resize if image is too large or too small
        height, width = img.shape[:2]
        target_width = 1000
        if width != target_width:
            scale = target_width / width
            img = cv2.resize(img, (target_width, int(height * scale)))
            
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Very wide color ranges to catch all Wordle tile variations
        # Green tiles
        green_lower = np.array([25, 20, 20])
        green_upper = np.array([95, 255, 255])
        
        # Yellow/Gold tiles
        yellow_lower = np.array([8, 60, 60])
        yellow_upper = np.array([45, 255, 255])
        
        # Gray tiles (very permissive)
        gray_lower = np.array([0, 0, 20])
        gray_upper = np.array([180, 100, 220])
        
        # Create masks for each color
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
        
        # Apply morphological operations to clean up
        kernel = np.ones((7,7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate expected cell size based on image dimensions
        expected_cell_size = width / 6  # Rough estimate
        min_area = (expected_cell_size * 0.2) ** 2  # Very lenient minimum
        max_area = (expected_cell_size * 3) ** 2    # Very lenient maximum
        
        cells = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Very lenient filter: reasonable size and aspect ratio
            if min_area < area < max_area and 0.4 < aspect_ratio < 2.0:
                # Determine color
                cell_roi_hsv = hsv[y:y+h, x:x+w]
                
                green_pixels = cv2.inRange(cell_roi_hsv, green_lower, green_upper).sum()
                yellow_pixels = cv2.inRange(cell_roi_hsv, yellow_lower, yellow_upper).sum()
                gray_pixels = cv2.inRange(cell_roi_hsv, gray_lower, gray_upper).sum()
                
                # Determine dominant color
                total = green_pixels + yellow_pixels + gray_pixels
                if total == 0:
                    continue
                    
                if green_pixels / total > 0.10:  # At least 10% green (more lenient)
                    color = 'green'
                elif yellow_pixels / total > 0.10:  # At least 10% yellow (more lenient)
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
        
        if len(cells) < 3:  # Need at least 3 cells (very lenient)
            return []
        
        # Sort cells by position
        cells.sort(key=lambda c: (c['y'], c['x']))
        
        # Group into rows using clustering
        rows = []
        current_row = []
        
        # Calculate average cell height
        avg_height = np.mean([c['h'] for c in cells])
        y_threshold = avg_height * 0.4  # 40% of cell height
        
        last_y = cells[0]['y'] if cells else 0
        
        for cell in cells:
            if abs(cell['y'] - last_y) > y_threshold and current_row:
                # New row
                if len(current_row) == 5:
                    current_row.sort(key=lambda c: c['x'])
                    rows.append(current_row)
                elif len(current_row) > 0:
                    # Try to pad the row if it's close to 5
                    current_row.sort(key=lambda c: c['x'])
                    if len(current_row) >= 3:  # At least 3 letters
                        rows.append(current_row)
                        
                current_row = [cell]
                last_y = cell['y']
            else:
                current_row.append(cell)
                last_y = max(last_y, cell['y'])
                
        # Don't forget the last row
        if len(current_row) == 5:
            current_row.sort(key=lambda c: c['x'])
            rows.append(current_row)
        elif len(current_row) >= 3:
            current_row.sort(key=lambda c: c['x'])
            rows.append(current_row)
                
        return rows
    
    def _extract_letter(self, cell_img: np.ndarray) -> Optional[str]:
        """Extract letter from cell image using OCR with error correction"""
        
        # Increase cell image size for better OCR
        scale_factor = 3
        cell_img = cv2.resize(cell_img, 
                             (cell_img.shape[1] * scale_factor, 
                              cell_img.shape[0] * scale_factor))
        
        # Convert to grayscale
        gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Try multiple preprocessing techniques
        techniques = []
        
        # 1. Simple binary threshold (inverted - white text on dark)
        _, thresh1 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
        techniques.append(thresh1)
        
        # 2. Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        techniques.append(thresh2)
        
        # 3. Otsu's threshold
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        techniques.append(thresh3)
        
        # 4. Try with morphological operations
        kernel = np.ones((2,2), np.uint8)
        thresh4 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        techniques.append(thresh4)
        
        # Try OCR on all techniques
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        best_letter = None
        best_confidence = 0
        
        for thresh in techniques:
            try:
                # Get detailed OCR data
                data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
                
                for i, text in enumerate(data['text']):
                    if text.strip():
                        conf = int(data['conf'][i])
                        letter = text.strip()[0].upper()
                        
                        # Apply corrections for common OCR mistakes
                        if letter in OCR_CORRECTIONS:
                            letter = OCR_CORRECTIONS[letter]
                        
                        if letter.isalpha() and conf > best_confidence:
                            best_letter = letter
                            best_confidence = conf
            except:
                # If detailed data fails, try simple string extraction
                try:
                    text = pytesseract.image_to_string(thresh, config=config).strip()
                    if text and len(text) >= 1:
                        letter = text[0].upper()
                        if letter in OCR_CORRECTIONS:
                            letter = OCR_CORRECTIONS[letter]
                        if letter.isalpha():
                            return letter
                except:
                    continue
        
        return best_letter
    
    def calculate_score(self, rows: List[List[Dict]]) -> Dict:
        """
        Calculate score based on letter appearances and conversions
        Returns dict with total score and letter breakdown
        """
        self.letter_history = {}
        letter_scores = []
        
        for row_idx, row in enumerate(rows, 1):
            if row_idx > 6:  # Max 6 rows in Wordle
                break
                
            for cell in row:
                letter = cell['letter']
                color = cell['color']
                
                # Skip gray letters (not in the word)
                if color == 'gray':
                    continue
                    
                # Track letter's first appearance
                if letter not in self.letter_history:
                    # First time seeing this letter
                    base_points = SCORING.get(row_idx, {'green': 4, 'yellow': 0})[color]
                    
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
                    'error': 'Could not detect Wordle grid in image. Please ensure the image clearly shows the Wordle game board with letters visible.'
                }
            
            # Add debug information
            debug_info = []
            for idx, row in enumerate(rows, 1):
                row_letters = [cell['letter'] for cell in row]
                row_colors = [cell['color'] for cell in row]
                debug_info.append({
                    'row': idx,
                    'letters': row_letters,
                    'colors': row_colors,
                    'count': len(row)
                })
                
            result = self.calculate_score(rows)
            result['debug'] = {
                'total_cells_detected': sum(len(row) for row in rows),
                'rows_detected': len(rows),
                'details': debug_info
            }
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
