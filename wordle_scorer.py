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

# Common 5-letter Wordle words (subset of most common ones)
VALID_WORDS = {
    'ABOUT', 'ABOVE', 'ACUTE', 'ADMIT', 'ADOPT', 'ADULT', 'AFTER', 'AGAIN', 'AGENT', 'AGREE',
    'AHEAD', 'ALARM', 'ALBUM', 'ALERT', 'ALIGN', 'ALIKE', 'ALIVE', 'ALLOW', 'ALONE', 'ALONG',
    'ALTER', 'ANGEL', 'ANGER', 'ANGLE', 'ANGRY', 'APART', 'APPLE', 'APPLY', 'ARENA', 'ARGUE',
    'ARISE', 'ARRAY', 'ASIDE', 'ASSET', 'AUDIO', 'AUDIT', 'AVOID', 'AWARD', 'AWARE', 'BADLY',
    'BAKER', 'BASES', 'BASIC', 'BASIS', 'BEACH', 'BEGAN', 'BEGIN', 'BEING', 'BELOW', 'BENCH',
    'BILLY', 'BIRTH', 'BLACK', 'BLAME', 'BLIND', 'BLOCK', 'BLOOD', 'BOARD', 'BOOST', 'BOOTH',
    'BOUND', 'BRAIN', 'BRAND', 'BREAD', 'BREAK', 'BREED', 'BRIEF', 'BRING', 'BROAD', 'BROKE',
    'BROWN', 'BUILD', 'BUILT', 'BUYER', 'CABLE', 'CALIF', 'CARRY', 'CATCH', 'CAUSE', 'CHAIN',
    'CHAIR', 'CHART', 'CHASE', 'CHEAP', 'CHECK', 'CHEST', 'CHIEF', 'CHILD', 'CHINA', 'CHOSE',
    'CIVIL', 'CLAIM', 'CLASS', 'CLEAN', 'CLEAR', 'CLICK', 'CLOCK', 'CLOSE', 'COACH', 'COAST',
    'COULD', 'COUNT', 'COURT', 'COVER', 'CRACK', 'CRAFT', 'CRASH', 'CRAZY', 'CREAM', 'CRIME',
    'CROSS', 'CROWD', 'CROWN', 'CRUDE', 'CYCLE', 'DAILY', 'DANCE', 'DATED', 'DEALT', 'DEATH',
    'DEBUT', 'DELAY', 'DELTA', 'DENSE', 'DEPTH', 'DOING', 'DOUBT', 'DOZEN', 'DRAFT', 'DRAMA',
    'DRANK', 'DRAWN', 'DREAM', 'DRESS', 'DRILL', 'DRINK', 'DRIVE', 'DROVE', 'DYING', 'EAGER',
    'EARLY', 'EARTH', 'EIGHT', 'ELECT', 'ELITE', 'EMPTY', 'ENEMY', 'ENJOY', 'ENTER', 'ENTRY',
    'EQUAL', 'ERROR', 'EVENT', 'EVERY', 'EXACT', 'EXIST', 'EXTRA', 'FAITH', 'FALSE', 'FAULT',
    'FIBER', 'FIELD', 'FIFTH', 'FIFTY', 'FIGHT', 'FINAL', 'FIRST', 'FIXED', 'FLASH', 'FLEET',
    'FLOOR', 'FLUID', 'FOCUS', 'FORCE', 'FORTH', 'FORTY', 'FORUM', 'FOUND', 'FRAME', 'FRANK',
    'FRAUD', 'FRESH', 'FRONT', 'FRUIT', 'FULLY', 'FUNNY', 'GIANT', 'GIVEN', 'GLASS', 'GLINT',
    'GLOBE', 'GOING', 'GRACE', 'GRADE', 'GRAND', 'GRANT', 'GRASS', 'GREAT', 'GREEN', 'GROSS',
    'GROUP', 'GROWN', 'GUARD', 'GUESS', 'GUEST', 'GUIDE', 'HAPPY', 'HARRY', 'HEART', 'HEAVY',
    'HENCE', 'HENRY', 'HORSE', 'HOTEL', 'HOUSE', 'HUMAN', 'IDEAL', 'IMAGE', 'INDEX', 'INNER',
    'INPUT', 'ISSUE', 'JAPAN', 'JIMMY', 'JOINT', 'JONES', 'JUDGE', 'KNOWN', 'LABEL', 'LARGE',
    'LASER', 'LATER', 'LAUGH', 'LAYER', 'LEARN', 'LEASE', 'LEAST', 'LEAVE', 'LEGAL', 'LEMON',
    'LEVEL', 'LEWIS', 'LIGHT', 'LIMIT', 'LINKS', 'LIVES', 'LOCAL', 'LOGIC', 'LOOSE', 'LOWER',
    'LUCKY', 'LUNCH', 'LYING', 'MAGIC', 'MAJOR', 'MAKER', 'MARCH', 'MARIA', 'MATCH', 'MAYBE',
    'MAYOR', 'MEANT', 'MEDIA', 'METAL', 'MIGHT', 'MINOR', 'MINUS', 'MIXED', 'MODEL', 'MONEY',
    'MONTH', 'MORAL', 'MOTOR', 'MOUNT', 'MOUSE', 'MOUTH', 'MOVIE', 'MUSIC', 'NEEDS', 'NEVER',
    'NEWLY', 'NIGHT', 'NOISE', 'NORTH', 'NOTED', 'NOVEL', 'NURSE', 'OCCUR', 'OCEAN', 'OFFER',
    'OFTEN', 'ORDER', 'OTHER', 'OUGHT', 'PAINT', 'PANEL', 'PAPER', 'PARTY', 'PEACE', 'PETER',
    'PHASE', 'PHONE', 'PHOTO', 'PIECE', 'PILOT', 'PITCH', 'PLACE', 'PLAIN', 'PLANE', 'PLANT',
    'PLATE', 'POINT', 'POUND', 'POWER', 'PRESS', 'PRICE', 'PRIDE', 'PRIME', 'PRINT', 'PRIOR',
    'PRISM', 'PRIZE', 'PROOF', 'PROUD', 'PROVE', 'QUEEN', 'QUICK', 'QUIET', 'QUITE', 'RADIO',
    'RAISE', 'RANGE', 'RAPID', 'RATIO', 'REACH', 'READY', 'REFER', 'RIGHT', 'RIVAL', 'RIVER',
    'ROBIN', 'ROCKY', 'ROGER', 'ROMAN', 'ROUGH', 'ROUND', 'ROUTE', 'ROYAL', 'RURAL', 'SCALE',
    'SCENE', 'SCOPE', 'SCORE', 'SENSE', 'SERVE', 'SEVEN', 'SHALL', 'SHAPE', 'SHARE', 'SHARP',
    'SHEET', 'SHELF', 'SHELL', 'SHIFT', 'SHINE', 'SHIRT', 'SHOCK', 'SHOOT', 'SHORT', 'SHOWN',
    'SIGHT', 'SINCE', 'SIXTH', 'SIXTY', 'SIZED', 'SKILL', 'SLEEP', 'SLIDE', 'SMALL', 'SMART',
    'SMILE', 'SMITH', 'SMOKE', 'SOLID', 'SOLVE', 'SORRY', 'SOUND', 'SOUTH', 'SPACE', 'SPARE',
    'SPEAK', 'SPEED', 'SPEND', 'SPENT', 'SPLIT', 'SPOKE', 'SPORT', 'STAFF', 'STAGE', 'STAKE',
    'STAND', 'START', 'STATE', 'STEAM', 'STEEL', 'STICK', 'STILL', 'STOCK', 'STONE', 'STOOD',
    'STORE', 'STORM', 'STORY', 'STRIP', 'STUCK', 'STUDY', 'STUFF', 'STYLE', 'SUGAR', 'SUITE',
    'SUPER', 'SWEET', 'TABLE', 'TAKEN', 'TASTE', 'TAXES', 'TEACH', 'TERRY', 'TEXAS', 'THANK',
    'THEFT', 'THEIR', 'THEME', 'THERE', 'THESE', 'THICK', 'THING', 'THINK', 'THIRD', 'THOSE',
    'THREE', 'THREW', 'THROW', 'TIGHT', 'TIMES', 'TITLE', 'TODAY', 'TOPIC', 'TOTAL', 'TOUCH',
    'TOUGH', 'TOWER', 'TRACK', 'TRADE', 'TRAIN', 'TRAIT', 'TREAT', 'TREND', 'TRIAL', 'TRIBE',
    'TRICK', 'TRIED', 'TRIES', 'TROOP', 'TRUCK', 'TRULY', 'TRUNK', 'TRUST', 'TRUTH', 'TWICE',
    'UNDER', 'UNDUE', 'UNION', 'UNITY', 'UNTIL', 'UPPER', 'URBAN', 'USAGE', 'USUAL', 'VALID',
    'VALUE', 'VIDEO', 'VIRUS', 'VISIT', 'VITAL', 'VOCAL', 'VOICE', 'WASTE', 'WATCH', 'WATER',
    'WHEEL', 'WHERE', 'WHICH', 'WHILE', 'WHITE', 'WHOLE', 'WHOSE', 'WOMAN', 'WOMEN', 'WORLD',
    'WORRY', 'WORSE', 'WORST', 'WORTH', 'WOULD', 'WOUND', 'WRITE', 'WRONG', 'WROTE', 'YOUNG',
    'YOUTH', 'CLOUT'
}

# Common OCR misreads for Wordle letters
OCR_CORRECTIONS = {
    '0': 'O',  # Zero to O
    '1': 'I',  # One to I
    '8': 'B',  # Eight to B
    '5': 'S',  # Five to S
    '6': 'G',  # Six to G
}

# Context-based corrections for similar looking letters
# When OCR is uncertain, these are common confusions in Wordle context
SIMILAR_LETTERS = {
    'O': ['D', 'Q', '0'],  # O can be confused with D, Q, or 0
    'I': ['L', '1', 'l'],  # I can be confused with L or 1
    'U': ['Y', 'V'],       # U can be confused with Y or V
    'Y': ['U', 'V'],       # Y can be confused with U or V
    'B': ['8', 'R'],       # B can be confused with 8 or R
    'S': ['5'],            # S can be confused with 5
}


class WordleScorer:
    """Analyzes Wordle game board images and calculates scores"""
    
    def __init__(self):
        self.letter_history = {}  # Track first appearance and conversions
        
    def _validate_and_correct_word(self, letters: List[str]) -> Tuple[List[str], bool]:
        """
        Validate if letters form a real word, and try corrections if not
        Returns: (corrected_letters, was_corrected)
        """
        if len(letters) != 5:
            return letters, False
            
        word = ''.join(letters).upper()
        
        # Check if it's already valid
        if word in VALID_WORDS:
            return letters, False
        
        # Try common substitutions for OCR errors
        substitutions = {
            'O': ['D', 'Q', '0'],
            'I': ['L', '1'],
            'U': ['V', 'Y'],
            'Y': ['V', 'U'],
            'S': ['5'],
            'B': ['8'],
        }
        
        # Try substituting each letter
        for i, letter in enumerate(letters):
            if letter in substitutions:
                for replacement in substitutions[letter]:
                    test_letters = letters.copy()
                    test_letters[i] = replacement
                    test_word = ''.join(test_letters).upper()
                    if test_word in VALID_WORDS:
                        return test_letters, True
        
        # Try reverse substitutions (D→O, L→I, etc.)
        reverse_subs = {
            'D': 'O', 'Q': 'O',
            'L': 'I', '1': 'I',
            'V': 'U',
            '5': 'S',
            '8': 'B'
        }
        
        for i, letter in enumerate(letters):
            if letter in reverse_subs:
                test_letters = letters.copy()
                test_letters[i] = reverse_subs[letter]
                test_word = ''.join(test_letters).upper()
                if test_word in VALID_WORDS:
                    return test_letters, True
        
        # No valid correction found
        return letters, False
    
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
        
        # POST-PROCESSING: Try to fix rows that don't have exactly 5 cells
        fixed_rows = []
        for row in rows:
            if len(row) == 5:
                fixed_rows.append(row)
            elif len(row) == 4:
                # Missing one cell - check for close duplicates
                # Look for cells that are very close together (might be duplicate letters)
                row_with_gap = self._try_find_duplicate_letter(row, avg_height)
                if len(row_with_gap) == 5:
                    fixed_rows.append(row_with_gap)
                else:
                    fixed_rows.append(row)  # Keep as-is even if incomplete
            elif len(row) > 5:
                # Too many cells - might have detected duplicates separately
                # Keep the 5 with best spacing
                fixed_rows.append(self._filter_to_five_cells(row))
            else:
                fixed_rows.append(row)  # Keep partial rows for now
                
        return fixed_rows
    
    def _try_find_duplicate_letter(self, row: List[Dict], avg_height: float) -> List[Dict]:
        """Try to find a missing duplicate letter by checking cell spacing"""
        # If we have 4 cells, there might be a duplicate that looks like one cell
        # Look for gaps in the X coordinates
        if len(row) < 4:
            return row
            
        x_positions = [c['x'] for c in row]
        avg_gap = (x_positions[-1] - x_positions[0]) / (len(row) - 1) if len(row) > 1 else avg_height
        
        # Check for a gap larger than 1.5x average
        for i in range(len(row) - 1):
            gap = row[i+1]['x'] - (row[i]['x'] + row[i]['w'])
            if gap > avg_gap * 1.5:
                # Found a large gap - duplicate the previous letter
                duplicate_cell = row[i].copy()
                duplicate_cell['x'] = row[i]['x'] + row[i]['w'] + 5
                row.insert(i+1, duplicate_cell)
                return row
                
        return row
    
    def _filter_to_five_cells(self, row: List[Dict]) -> List[Dict]:
        """If row has more than 5 cells, keep the 5 with best spacing"""
        if len(row) <= 5:
            return row
            
        # Keep cells with most uniform spacing
        # Simple heuristic: remove cells that are very close to others
        filtered = []
        min_distance = 10  # Minimum pixels between cells
        
        for cell in row:
            too_close = False
            for existing in filtered:
                if abs(cell['x'] - existing['x']) < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered.append(cell)
                
        return filtered[:5]  # Take first 5
    
    def _extract_letter(self, cell_img: np.ndarray) -> Optional[str]:
        """Extract letter from cell image using OCR with error correction"""
        
        # Increase cell image size for better OCR
        scale_factor = 4  # Increased from 3
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
        
        # STRICT: Only uppercase letters A-Z
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        best_letter = None
        best_confidence = 0
        
        for thresh in techniques:
            try:
                # Get detailed OCR data with confidence
                data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
                
                for i, text in enumerate(data['text']):
                    if text.strip():
                        conf = int(data['conf'][i])
                        letter = text.strip()[0].upper()
                        
                        # Only accept valid uppercase letters
                        if not letter.isalpha() or not letter.isupper():
                            continue
                        
                        # Apply corrections for common OCR mistakes
                        if letter in OCR_CORRECTIONS:
                            letter = OCR_CORRECTIONS[letter]
                        
                        if conf > best_confidence:
                            best_letter = letter
                            best_confidence = conf
            except:
                # If detailed data fails, try simple string extraction
                try:
                    text = pytesseract.image_to_string(thresh, config=config).strip()
                    if text and len(text) >= 1:
                        letter = text[0].upper()
                        
                        # Only accept valid uppercase letters
                        if not letter.isalpha() or not letter.isupper():
                            continue
                            
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
            
            # Add debug information and word validation
            debug_info = []
            corrections_made = []
            
            for idx, row in enumerate(rows, 1):
                row_letters = [cell['letter'] for cell in row]
                row_colors = [cell['color'] for cell in row]
                original_word = ''.join(row_letters)
                
                # Try to validate and correct the word
                if len(row_letters) == 5:
                    corrected_letters, was_corrected = self._validate_and_correct_word(row_letters)
                    if was_corrected:
                        corrected_word = ''.join(corrected_letters)
                        corrections_made.append(f"Row {idx}: {original_word} → {corrected_word}")
                        # Update the cells with corrected letters
                        for i, cell in enumerate(row):
                            cell['letter'] = corrected_letters[i]
                        row_letters = corrected_letters
                
                debug_info.append({
                    'row': idx,
                    'letters': row_letters,
                    'colors': row_colors,
                    'count': len(row),
                    'word': ''.join(row_letters),
                    'valid': ''.join(row_letters).upper() in VALID_WORDS
                })
                
            result = self.calculate_score(rows)
            result['debug'] = {
                'total_cells_detected': sum(len(row) for row in rows),
                'rows_detected': len(rows),
                'details': debug_info,
                'corrections': corrections_made if corrections_made else None
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
