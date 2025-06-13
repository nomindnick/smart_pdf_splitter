"""OCR post-processing module for text cleanup and correction."""

import re
import string
from typing import Dict, List, Tuple, Optional, Any
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class OCRPostProcessor:
    """Clean and enhance OCR output text."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize post-processor with language-specific rules.
        
        Args:
            language: Language code for processing rules
        """
        self.language = language
        
        # Common OCR substitution errors
        self.ocr_corrections = [
            # Visual similarity errors
            (r'\brn\b', 'm'),  # rn -> m
            (r'\bcl\b', 'd'),  # cl -> d
            (r'\bvv\b', 'w'),  # vv -> w
            (r'\bVV\b', 'W'),  # VV -> W
            
            # Letter/number confusion
            (r'(?<![a-zA-Z])O(?![a-zA-Z])', '0'),  # O -> 0 (isolated)
            (r'(?<![0-9])0(?![0-9])(?=[a-zA-Z])', 'O'),  # 0 -> O (in words)
            (r'(?<![a-zA-Z])l(?![a-zA-Z])', '1'),  # l -> 1 (isolated)
            (r'(?<![0-9])1(?![0-9])(?=[a-zA-Z])', 'I'),  # 1 -> I (in words)
            (r'\bI1\b', 'II'),  # I1 -> II
            (r'\b11\b', 'II'),  # 11 -> II (Roman numeral context)
            
            # Common word corrections
            (r'\btl1e\b', 'the'),
            (r'\btI1e\b', 'the'),
            (r'\bth(?:e|c)\b', 'the'),
            (r'\band\b', 'and'),
            (r'\bof\b', 'of'),
            (r'\bto\b', 'to'),
            
            # Punctuation cleanup
            (r'(?<=[a-zA-Z])(?=[.!?,;:])', ' '),  # Add space before punctuation
            (r'(?<=[.!?])(?=[A-Z])', ' '),  # Add space after sentence end
            (r'\s+([.!?,;:])', r'\1'),  # Remove space before punctuation
            (r'([.!?])\s*\n\s*([a-z])', r'\1 \2'),  # Fix sentence continuation
            
            # Remove OCR artifacts
            (r'[|]{2,}', ''),  # Multiple pipes
            (r'[~]{2,}', '-'),  # Multiple tildes to dash
            (r'[-]{3,}', '—'),  # Multiple dashes to em dash
            (r'[_]{3,}', ''),  # Multiple underscores
            (r'#+\s*#+', ''),  # Hash patterns
        ]
        
        # Domain-specific corrections
        self.domain_corrections = {
            "email": [
                (r'@\s+', '@'),  # Remove space after @
                (r'\s+@', '@'),  # Remove space before @
                (r'\.\s+com\b', '.com'),
                (r'\.\s+org\b', '.org'),
                (r'\.\s+net\b', '.net'),
                (r'\bgrnail\b', 'gmail'),
                (r'\byahoo\b', 'yahoo'),
            ],
            "invoice": [
                (r'\$\s+', '$'),  # Remove space after $
                (r'#\s+', '#'),  # Remove space after #
                (r'\bInv(?:o|0)ice\b', 'Invoice'),
                (r'\bP(?:O|0)\b', 'PO'),  # Purchase Order
                (r'\bQty\b', 'Qty'),
            ],
            "contract": [
                (r'\bSECTION\b', 'SECTION'),
                (r'\bARTICLE\b', 'ARTICLE'),
                (r'\bCLAUSE\b', 'CLAUSE'),
                (r'\bWHEREAS\b', 'WHEREAS'),
                (r'\bTHEREFORE\b', 'THEREFORE'),
            ]
        }
        
        # Common abbreviations that shouldn't have periods added
        self.abbreviations = {
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
            "co", "corp", "inc", "ltd", "llc",
            "st", "ave", "blvd", "rd",
            "vs", "etc", "eg", "ie"
        }
    
    def process_text(
        self,
        text: str,
        document_type: Optional[str] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        apply_aggressive: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply post-processing to clean OCR text.
        
        Args:
            text: Raw OCR text
            document_type: Type of document for specific rules
            confidence_scores: OCR confidence scores to guide processing
            apply_aggressive: Apply more aggressive corrections
            
        Returns:
            Tuple of (processed_text, processing_info)
        """
        if not text:
            return "", {"corrections_made": 0, "steps_applied": []}
        
        info = {
            "corrections_made": 0,
            "steps_applied": [],
            "specific_corrections": []
        }
        
        original_text = text
        
        # Step 1: Basic cleanup
        text, basic_corrections = self._basic_cleanup(text)
        info["corrections_made"] += basic_corrections
        if basic_corrections > 0:
            info["steps_applied"].append("basic_cleanup")
        
        # Step 2: Apply OCR corrections
        text, ocr_corrections = self._apply_ocr_corrections(text)
        info["corrections_made"] += ocr_corrections
        if ocr_corrections > 0:
            info["steps_applied"].append("ocr_corrections")
            
        # Step 3: Fix word boundaries
        text, boundary_fixes = self._fix_word_boundaries(text)
        info["corrections_made"] += boundary_fixes
        if boundary_fixes > 0:
            info["steps_applied"].append("word_boundaries")
        
        # Step 4: Apply domain-specific corrections
        if document_type and document_type in self.domain_corrections:
            text, domain_corrections = self._apply_domain_corrections(text, document_type)
            info["corrections_made"] += domain_corrections
            if domain_corrections > 0:
                info["steps_applied"].append(f"{document_type}_corrections")
        
        # Step 5: Fix spacing and formatting
        text, format_corrections = self._fix_formatting(text)
        info["corrections_made"] += format_corrections
        if format_corrections > 0:
            info["steps_applied"].append("formatting")
        
        # Step 6: Aggressive corrections if confidence is low
        if apply_aggressive or (confidence_scores and confidence_scores.get("overall_confidence", 1) < 0.6):
            text, aggressive_corrections = self._apply_aggressive_corrections(text)
            info["corrections_made"] += aggressive_corrections
            if aggressive_corrections > 0:
                info["steps_applied"].append("aggressive_corrections")
        
        # Step 7: Final cleanup
        text = self._final_cleanup(text)
        
        # Calculate improvement
        info["text_changed"] = text != original_text
        if info["text_changed"]:
            similarity = SequenceMatcher(None, original_text, text).ratio()
            info["similarity_ratio"] = round(similarity, 3)
        
        return text, info
    
    def _basic_cleanup(self, text: str) -> Tuple[str, int]:
        """Basic text cleanup."""
        corrections = 0
        original = text
        
        # Remove null characters and control characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        if text != original:
            corrections += 1
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text, corrections
    
    def _apply_ocr_corrections(self, text: str) -> Tuple[str, int]:
        """Apply common OCR error corrections."""
        corrections = 0
        
        for pattern, replacement in self.ocr_corrections:
            new_text = re.sub(pattern, replacement, text)
            if new_text != text:
                corrections += len(re.findall(pattern, text))
                text = new_text
        
        return text, corrections
    
    def _fix_word_boundaries(self, text: str) -> Tuple[str, int]:
        """Fix word boundary issues."""
        corrections = 0
        
        # Split merged words (simple heuristic)
        # Look for lowercase followed by uppercase without space
        pattern = r'([a-z])([A-Z])'
        new_text = re.sub(pattern, r'\1 \2', text)
        if new_text != text:
            corrections += len(re.findall(pattern, text))
            text = new_text
        
        # Fix words split by line breaks
        pattern = r'([a-z])-\n([a-z])'
        new_text = re.sub(pattern, r'\1\2', text)
        if new_text != text:
            corrections += len(re.findall(pattern, text))
            text = new_text
        
        return text, corrections
    
    def _apply_domain_corrections(self, text: str, document_type: str) -> Tuple[str, int]:
        """Apply domain-specific corrections."""
        corrections = 0
        
        if document_type in self.domain_corrections:
            for pattern, replacement in self.domain_corrections[document_type]:
                new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if new_text != text:
                    corrections += len(re.findall(pattern, text, re.IGNORECASE))
                    text = new_text
        
        return text, corrections
    
    def _fix_formatting(self, text: str) -> Tuple[str, int]:
        """Fix spacing and formatting issues."""
        corrections = 0
        original = text
        
        # Fix spacing around punctuation
        # Add space after punctuation if missing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', text)
        
        # Remove extra spaces before punctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        
        # Fix quotes
        text = re.sub(r'"\s*([^"]+?)\s*"', r'"\1"', text)
        text = re.sub(r"'\s*([^']+?)\s*'", r"'\1'", text)
        
        # Fix parentheses
        text = re.sub(r'\(\s*', '(', text)
        text = re.sub(r'\s*\)', ')', text)
        
        # Fix dashes
        text = re.sub(r'\s*-\s*', '-', text)
        text = re.sub(r'--', '—', text)
        
        if text != original:
            corrections = 1  # Count as one formatting correction
        
        return text, corrections
    
    def _apply_aggressive_corrections(self, text: str) -> Tuple[str, int]:
        """Apply more aggressive corrections for low-confidence text."""
        corrections = 0
        
        # Fix common word fragments
        word_corrections = [
            (r'\bth\s+e\b', 'the'),
            (r'\ba\s+nd\b', 'and'),
            (r'\bt\s+o\b', 'to'),
            (r'\bo\s+f\b', 'of'),
            (r'\bi\s+n\b', 'in'),
            (r'\bf\s+or\b', 'for'),
            (r'\bw\s+ith\b', 'with'),
            (r'\bt\s+hat\b', 'that'),
            (r'\bt\s+his\b', 'this'),
        ]
        
        for pattern, replacement in word_corrections:
            new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new_text != text:
                corrections += len(re.findall(pattern, text, re.IGNORECASE))
                text = new_text
        
        # Fix sentence capitalization
        sentences = re.split(r'([.!?]\s+)', text)
        fixed_sentences = []
        
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Text parts (not delimiters)
                # Capitalize first letter of sentence
                if part and part[0].islower():
                    part = part[0].upper() + part[1:]
                    corrections += 1
                fixed_sentences.append(part)
            else:
                fixed_sentences.append(part)
        
        text = ''.join(fixed_sentences)
        
        return text, corrections
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass."""
        # Remove any remaining multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove spaces at start/end of lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove trailing whitespace
        text = text.strip()
        
        return text
    
    def get_correction_suggestions(
        self,
        text: str,
        confidence_scores: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get specific correction suggestions without applying them.
        
        Args:
            text: OCR text to analyze
            confidence_scores: Confidence scores from OCR
            
        Returns:
            List of correction suggestions
        """
        suggestions = []
        
        # Check for common OCR patterns
        for pattern, replacement in self.ocr_corrections[:10]:  # First 10 most common
            matches = list(re.finditer(pattern, text))
            if matches:
                for match in matches[:3]:  # Limit to 3 examples per pattern
                    suggestions.append({
                        "type": "ocr_error",
                        "position": match.start(),
                        "original": match.group(),
                        "suggestion": replacement,
                        "context": text[max(0, match.start()-20):match.end()+20]
                    })
        
        # Check for formatting issues
        # Missing spaces after punctuation
        missing_spaces = list(re.finditer(r'([.!?])([A-Z])', text))
        for match in missing_spaces[:3]:
            suggestions.append({
                "type": "formatting",
                "position": match.start(),
                "original": match.group(),
                "suggestion": match.group(1) + ' ' + match.group(2),
                "description": "Missing space after punctuation"
            })
        
        # Check for word boundary issues
        boundary_issues = list(re.finditer(r'([a-z])([A-Z])', text))
        for match in boundary_issues[:3]:
            suggestions.append({
                "type": "word_boundary",
                "position": match.start(),
                "original": match.group(),
                "suggestion": match.group(1) + ' ' + match.group(2),
                "description": "Possible merged words"
            })
        
        return suggestions