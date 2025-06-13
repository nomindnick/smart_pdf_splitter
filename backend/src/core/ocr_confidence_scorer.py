"""OCR confidence scoring system for quality assessment."""

import re
import string
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class OCRConfidenceScorer:
    """Score OCR output quality and confidence."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize with language-specific resources.
        
        Args:
            language: Language code for validation
        """
        self.language = language
        
        # Load common words for the language
        self.common_words = self._load_common_words()
        
        # OCR error patterns with their corrections
        self.ocr_error_patterns = [
            (r'\brn\b', 'm'),  # rn -> m
            (r'\bcl\b', 'd'),  # cl -> d
            (r'\bl\b(?=[a-z])', 'I'),  # l -> I at word start
            (r'(?<![a-zA-Z])O(?![a-zA-Z])', '0'),  # O -> 0 when not in word
            (r'(?<![a-zA-Z])0(?![0-9])', 'O'),  # 0 -> O when not in number
            (r'\b11\b', 'II'),  # 11 -> II
            (r'[|]{2,}', ''),  # Multiple pipes (remove)
            (r'[~]{2,}', '-'),  # Multiple tildes to dash
        ]
        
        # Pattern validators
        self.validators = {
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "url": re.compile(r'https?://(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            "phone": re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'),
            "date": [
                re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}'),
                re.compile(r'\d{4}-\d{2}-\d{2}'),
                re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}')
            ],
            "currency": re.compile(r'[$€£¥]\s*\d+[,.\d]*'),
            "percentage": re.compile(r'\d+\.?\d*\s*%'),
        }
    
    def score_ocr_output(
        self,
        text: str,
        expected_language: Optional[str] = None,
        document_type: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence scores for OCR output.
        
        Args:
            text: OCR output text
            expected_language: Expected language code
            document_type: Type of document
            page_number: Page number in document
            
        Returns:
            Dictionary with confidence scores and analysis
        """
        if not text:
            return {
                "overall_confidence": 0.0,
                "quality_assessment": "empty",
                "issues": ["no_text_detected"],
                "details": {}
            }
        
        # Use expected language or default
        lang = expected_language or self.language
        
        # Calculate component scores
        scores = {
            "character_confidence": self._character_level_confidence(text),
            "word_confidence": self._word_level_confidence(text),
            "structure_confidence": self._structure_confidence(text),
            "language_confidence": self._language_confidence(text, lang),
        }
        
        # Add document-specific confidence if type is provided
        if document_type:
            scores[f"{document_type}_confidence"] = self._document_specific_confidence(
                text, document_type
            )
        
        # Calculate pattern scores
        pattern_scores = self._pattern_confidence(text)
        scores.update(pattern_scores)
        
        # Calculate overall confidence
        weights = {
            "character_confidence": 0.25,
            "word_confidence": 0.30,
            "structure_confidence": 0.20,
            "language_confidence": 0.25
        }
        
        overall = sum(
            scores.get(key, 0) * weight
            for key, weight in weights.items()
        )
        
        # Prepare result
        result = {
            "overall_confidence": round(overall, 3),
            "quality_assessment": self._assess_quality(overall),
            "scores": scores,
            "issues": self._identify_issues(text, scores),
            "suggestions": [],
            "metrics": self._calculate_metrics(text),
        }
        
        # Add suggestions based on issues
        result["suggestions"] = self._generate_suggestions(result["issues"], scores)
        
        # Add page info if provided
        if page_number is not None:
            result["page_number"] = page_number
        
        return result
    
    def _load_common_words(self) -> set:
        """Load common words for the language."""
        # For now, use a basic English word list
        # In production, load from a file
        if self.language == "en":
            return {
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                "or", "an", "will", "my", "one", "all", "would", "there", "their",
                "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
                "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
                "take", "people", "into", "year", "your", "good", "some", "could", "them",
                "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
                "think", "also", "back", "after", "use", "two", "how", "our", "work",
                "first", "well", "way", "even", "new", "want", "because", "any", "these",
                "give", "day", "most", "us", "is", "was", "are", "been", "has", "had",
                "were", "been", "have", "their", "said", "each", "she", "which", "do",
                "their", "time", "if", "will", "way", "about", "many", "then", "them",
                "write", "would", "like", "so", "these", "her", "long", "make", "thing",
                "see", "him", "two", "has", "look", "more", "day", "could", "go", "come"
            }
        return set()
    
    def _character_level_confidence(self, text: str) -> float:
        """Assess character-level quality."""
        if not text:
            return 0.0
        
        total_chars = len(text)
        
        # Character type distribution
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        spaces = sum(1 for c in text if c.isspace())
        punctuation = sum(1 for c in text if c in string.punctuation)
        special = total_chars - letters - digits - spaces - punctuation
        
        # Calculate ratios
        letter_ratio = letters / total_chars
        special_ratio = special / total_chars
        
        # Check for suspicious patterns
        suspicious_count = 0
        
        # Non-printable characters
        non_printable = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        suspicious_count += non_printable
        
        # Repeated special characters
        for pattern in [r'[|]{3,}', r'[~]{3,}', r'[#]{5,}', r'[@]{3,}']:
            suspicious_count += len(re.findall(pattern, text))
        
        # Calculate confidence
        confidence = 1.0
        
        # Penalize special characters
        if special_ratio > 0.1:
            confidence -= min(special_ratio * 2, 0.5)
        
        # Penalize suspicious patterns
        if suspicious_count > 0:
            confidence -= min(suspicious_count / total_chars * 10, 0.5)
        
        # Reward normal distribution
        if 0.6 <= letter_ratio <= 0.85:
            confidence = min(confidence + 0.1, 1.0)
        
        return max(0.0, confidence)
    
    def _word_level_confidence(self, text: str) -> float:
        """Assess word-level quality."""
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return 0.0
        
        total_words = len(words)
        
        # Score components
        valid_score = 0
        length_score = 0
        repetition_score = 1.0
        
        # Check each word
        for word in words:
            # Common word check
            if word in self.common_words:
                valid_score += 1
            # Reasonable length check
            elif 2 <= len(word) <= 15:
                valid_score += 0.3
                length_score += 1
            # Very long words are suspicious
            elif len(word) > 20:
                valid_score -= 0.2
        
        # Calculate base confidence
        confidence = valid_score / total_words if total_words > 0 else 0
        
        # Check for repetition
        word_freq = Counter(words)
        if word_freq:
            most_common_freq = max(word_freq.values())
            if most_common_freq > total_words * 0.2:  # Same word > 20%
                repetition_score = 0.7
        
        # Check for gibberish patterns
        gibberish_count = sum(
            1 for word in words
            if len(word) > 5 and self._is_gibberish(word)
        )
        
        if gibberish_count > total_words * 0.1:
            confidence *= 0.8
        
        return min(1.0, confidence * repetition_score)
    
    def _is_gibberish(self, word: str) -> bool:
        """Check if a word appears to be gibberish."""
        # Check for too many consonants in a row
        consonant_groups = re.findall(r'[bcdfghjklmnpqrstvwxyz]{4,}', word.lower())
        if consonant_groups:
            return True
        
        # Check for unusual character repetition
        for i in range(len(word) - 2):
            if word[i] == word[i+1] == word[i+2]:
                return True
        
        # Check vowel ratio
        vowels = sum(1 for c in word.lower() if c in 'aeiou')
        if len(word) > 4 and vowels / len(word) < 0.2:
            return True
        
        return False
    
    def _structure_confidence(self, text: str) -> float:
        """Assess structural quality."""
        if not text:
            return 0.0
        
        scores = []
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Check capitalization
                if sentence and sentence[0].isupper():
                    valid_sentences += 0.3
                
                # Check word count
                word_count = len(sentence.split())
                if 3 <= word_count <= 50:
                    valid_sentences += 0.7
        
        if sentences:
            sentence_score = min(valid_sentences / len(sentences), 1.0)
            scores.append(sentence_score)
        
        # Whitespace quality
        whitespace_score = 1.0
        
        # Check for excessive spaces
        if '   ' in text:  # Triple spaces
            whitespace_score -= 0.2
        
        # Check for missing spaces after punctuation
        missing_spaces = len(re.findall(r'[.!?,][a-zA-Z]', text))
        if missing_spaces > 5:
            whitespace_score -= 0.3
        
        scores.append(max(0, whitespace_score))
        
        # Line structure
        lines = text.split('\n')
        empty_ratio = sum(1 for line in lines if not line.strip()) / len(lines)
        
        if empty_ratio > 0.5:
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _language_confidence(self, text: str, expected_language: str) -> float:
        """Assess language match confidence."""
        if expected_language != "en":
            # For non-English, return neutral score
            return 0.5
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return 0.0
        
        # Count English indicators
        english_score = 0
        total_checks = 0
        
        # Common English words
        common_word_count = sum(1 for word in words if word in self.common_words)
        if words:
            english_score += common_word_count / len(words)
            total_checks += 1
        
        # English patterns
        patterns = [
            r'\bthe\s+\w+',
            r'\band\s+\w+',
            r'\bof\s+\w+',
            r'\bto\s+\w+',
            r'\bin\s+\w+',
            r'\bis\s+\w+',
            r'\bwas\s+\w+',
        ]
        
        pattern_count = sum(
            len(re.findall(pattern, text.lower()))
            for pattern in patterns
        )
        
        if pattern_count > 0:
            english_score += min(pattern_count / 20, 1.0)
            total_checks += 1
        
        return english_score / total_checks if total_checks > 0 else 0.0
    
    def _pattern_confidence(self, text: str) -> Dict[str, float]:
        """Check for valid patterns in text."""
        scores = {}
        
        # Email detection
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if emails:
            valid_emails = sum(1 for email in emails if '@' in email and '.' in email)
            scores["email_pattern_confidence"] = valid_emails / len(emails)
        
        # Date detection
        date_count = 0
        for pattern in self.validators["date"]:
            date_count += len(pattern.findall(text))
        
        if date_count > 0:
            scores["date_pattern_confidence"] = min(date_count / 5, 1.0)
        
        # Currency detection
        currency_matches = self.validators["currency"].findall(text)
        if currency_matches:
            scores["currency_pattern_confidence"] = min(len(currency_matches) / 3, 1.0)
        
        return scores
    
    def _document_specific_confidence(self, text: str, document_type: str) -> float:
        """Calculate confidence for specific document types."""
        if document_type == "email":
            return self._email_confidence(text)
        elif document_type == "invoice":
            return self._invoice_confidence(text)
        elif document_type == "contract":
            return self._contract_confidence(text)
        else:
            return 0.5  # Neutral score for unknown types
    
    def _email_confidence(self, text: str) -> float:
        """Calculate email-specific confidence."""
        scores = []
        
        # Check headers
        headers = ["from:", "to:", "subject:", "date:"]
        found_headers = sum(
            1 for header in headers
            if re.search(f'^{header}', text, re.MULTILINE | re.IGNORECASE)
        )
        scores.append(found_headers / len(headers))
        
        # Check for email addresses
        email_count = len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text))
        scores.append(min(email_count / 2, 1.0))
        
        # Check for typical email phrases
        email_phrases = [
            "dear", "regards", "sincerely", "thank you", "please", "attached"
        ]
        phrase_count = sum(
            1 for phrase in email_phrases
            if phrase in text.lower()
        )
        scores.append(min(phrase_count / 3, 1.0))
        
        return sum(scores) / len(scores)
    
    def _invoice_confidence(self, text: str) -> float:
        """Calculate invoice-specific confidence."""
        scores = []
        
        # Check for invoice keywords
        keywords = ["invoice", "bill", "total", "amount", "payment", "due"]
        keyword_count = sum(
            1 for keyword in keywords
            if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE)
        )
        scores.append(min(keyword_count / 3, 1.0))
        
        # Check for monetary amounts
        amount_pattern = r'[$€£¥]\s*[\d,]+\.?\d*'
        amounts = re.findall(amount_pattern, text)
        scores.append(min(len(amounts) / 3, 1.0))
        
        # Check for invoice number
        invoice_num = re.search(r'invoice\s*#?\s*:?\s*\d+', text, re.IGNORECASE)
        scores.append(1.0 if invoice_num else 0.0)
        
        return sum(scores) / len(scores)
    
    def _contract_confidence(self, text: str) -> float:
        """Calculate contract-specific confidence."""
        scores = []
        
        # Legal terms
        legal_terms = [
            "agreement", "party", "parties", "whereas", "therefore",
            "shall", "hereby", "terms", "conditions"
        ]
        term_count = sum(
            1 for term in legal_terms
            if re.search(r'\b' + term + r'\b', text, re.IGNORECASE)
        )
        scores.append(min(term_count / 4, 1.0))
        
        # Section markers
        section_pattern = r'(?:section|article|clause)\s+\d+'
        sections = re.findall(section_pattern, text, re.IGNORECASE)
        scores.append(min(len(sections) / 3, 1.0))
        
        return sum(scores) / len(scores)
    
    def _assess_quality(self, confidence: float) -> str:
        """Convert confidence score to quality assessment."""
        if confidence >= 0.9:
            return "excellent"
        elif confidence >= 0.8:
            return "good"
        elif confidence >= 0.6:
            return "fair"
        elif confidence >= 0.4:
            return "poor"
        else:
            return "very_poor"
    
    def _identify_issues(self, text: str, scores: Dict[str, float]) -> List[str]:
        """Identify specific issues in OCR output."""
        issues = []
        
        # Score-based issues
        if scores.get("character_confidence", 1) < 0.7:
            issues.append("suspicious_characters")
        
        if scores.get("word_confidence", 1) < 0.6:
            issues.append("low_word_recognition")
        
        if scores.get("structure_confidence", 1) < 0.6:
            issues.append("poor_text_structure")
        
        if scores.get("language_confidence", 1) < 0.5:
            issues.append("language_detection_failed")
        
        # Pattern-based issues
        if re.search(r'[|~#@]{3,}', text):
            issues.append("ocr_artifacts_detected")
        
        if re.search(r'\b\w{25,}\b', text):
            issues.append("word_boundary_errors")
        
        if text.count('\n\n\n') > 3:
            issues.append("excessive_line_breaks")
        
        # Check for common OCR errors
        ocr_errors = [
            (r'\brn\b', "rn_to_m_errors"),
            (r'(?<!\w)[0O](?!\w)', "zero_o_confusion"),
            (r'\b[Il1]{2,}\b', "i_l_1_confusion")
        ]
        
        for pattern, issue in ocr_errors:
            if re.search(pattern, text):
                issues.append(issue)
        
        return issues
    
    def _generate_suggestions(self, issues: List[str], scores: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        
        if "suspicious_characters" in issues:
            suggestions.append("Improve scan quality or increase image resolution")
        
        if "low_word_recognition" in issues:
            suggestions.append("Check OCR language settings match document language")
        
        if "poor_text_structure" in issues:
            suggestions.append("Ensure proper page layout analysis before OCR")
        
        if "ocr_artifacts_detected" in issues:
            suggestions.append("Apply image preprocessing (deskew, denoise, contrast enhancement)")
        
        if "word_boundary_errors" in issues:
            suggestions.append("Adjust OCR segmentation parameters")
        
        if scores.get("overall_confidence", 0) < 0.5:
            suggestions.append("Consider manual review or re-scanning at higher quality")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions
    
    def _calculate_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate text metrics."""
        words = text.split()
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": len(text.split('\n')),
            "average_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "unique_words": len(set(w.lower() for w in words))
        }