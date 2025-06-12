"""
Optimized prompts for phi4-mini:3.8b boundary detection.

This module contains carefully crafted prompts that work well with
phi4-mini's capabilities and limitations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PromptStrategy(str, Enum):
    """Different prompt strategies for various scenarios."""
    
    CONCISE = "concise"          # Minimal tokens, fast
    DETAILED = "detailed"        # More context, slower
    FEW_SHOT = "few_shot"       # Include examples
    CHAIN_OF_THOUGHT = "cot"    # Step-by-step reasoning
    STRUCTURED = "structured"    # JSON output focus


@dataclass
class PromptTemplate:
    """Template for boundary detection prompts."""
    
    name: str
    strategy: PromptStrategy
    system_instruction: str
    user_template: str
    examples: Optional[List[Dict[str, str]]] = None
    max_tokens: int = 200
    

class Phi4MiniPromptLibrary:
    """Library of optimized prompts for phi4-mini boundary detection."""
    
    # Base templates for different strategies
    TEMPLATES = {
        "concise_general": PromptTemplate(
            name="concise_general",
            strategy=PromptStrategy.CONCISE,
            system_instruction="Detect document boundaries. Be concise.",
            user_template="""Check if these are different documents:

PREV: {prev_text}
CURR: {curr_text}

JSON: {{"boundary": bool, "confidence": 0-1, "type": str, "reason": str}}""",
            max_tokens=150
        ),
        
        "detailed_construction": PromptTemplate(
            name="detailed_construction",
            strategy=PromptStrategy.DETAILED,
            system_instruction="You are an expert in construction documents (RFIs, submittals, change orders).",
            user_template="""Analyze these pages for document boundaries in construction context:

PREVIOUS PAGE (ending):
{prev_text}

CURRENT PAGE (beginning):
{curr_text}

Consider:
- Construction document types (RFI, submittal, change order, daily report)
- Document numbers and references
- Project information changes
- Formal document headers

Respond in JSON:
{{
    "boundary": true/false,
    "confidence": 0.0-1.0,
    "type": "rfi|submittal|change_order|invoice|letter|report|other",
    "reason": "specific indicators found"
}}""",
            max_tokens=250
        ),
        
        "few_shot_email": PromptTemplate(
            name="few_shot_email",
            strategy=PromptStrategy.FEW_SHOT,
            system_instruction="Detect email boundaries using these patterns.",
            user_template="""Examples of email boundaries:
1. "...Regards, John" → "From: jane@co.com" = boundary (letter→email)
2. "...attachment below" → "Page 2 of email" = no boundary (continuation)

Now analyze:
PREV: {prev_text}
CURR: {curr_text}

JSON: {{"boundary": bool, "confidence": 0-1, "type": str, "reason": str}}""",
            examples=[
                {
                    "prev": "Best regards,\nJohn Smith\nProject Manager",
                    "curr": "From: alice@example.com\nTo: team@example.com\nSubject: Update",
                    "result": '{"boundary": true, "confidence": 0.95, "type": "email", "reason": "Letter closing followed by email header"}'
                }
            ],
            max_tokens=200
        ),
        
        "cot_financial": PromptTemplate(
            name="cot_financial",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            system_instruction="Analyze financial documents step by step.",
            user_template="""Analyze for financial document boundaries:

PREVIOUS: {prev_text}
CURRENT: {curr_text}

Steps:
1. Check if PREVIOUS ends a document (totals, signatures, page X of X)
2. Check if CURRENT starts new document (invoice #, date, header)
3. Compare document types and numbers

JSON: {{"boundary": bool, "confidence": 0-1, "type": "invoice|po|quote|statement|other", "reason": str}}""",
            max_tokens=300
        ),
        
        "structured_mixed": PromptTemplate(
            name="structured_mixed",
            strategy=PromptStrategy.STRUCTURED,
            system_instruction="Output only valid JSON for document boundary detection.",
            user_template="""PREV_END: {prev_text}
CURR_START: {curr_text}
PATTERNS: {patterns}

{{
  "boundary": boolean,
  "confidence": number (0-1),
  "type": "email|invoice|contract|letter|rfi|submittal|report|other",
  "reason": "string (max 50 chars)"
}}""",
            max_tokens=150
        )
    }
    
    @classmethod
    def get_prompt(
        cls,
        template_name: str,
        prev_text: str,
        curr_text: str,
        patterns: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Get a formatted prompt from template."""
        
        template = cls.TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
            
        # Format the prompt
        format_args = {
            "prev_text": cls._truncate_text(prev_text, 400),
            "curr_text": cls._truncate_text(curr_text, 400),
            "patterns": ", ".join(patterns) if patterns else "none",
            **kwargs
        }
        
        prompt = template.user_template.format(**format_args)
        
        # Add system instruction if needed
        if template.system_instruction:
            prompt = f"{template.system_instruction}\n\n{prompt}"
            
        return prompt
    
    @classmethod
    def get_optimized_prompt(
        cls,
        prev_text: str,
        curr_text: str,
        detected_patterns: List[str] = None,
        document_focus: str = "general",
        use_examples: bool = True
    ) -> str:
        """
        Get an optimized prompt based on context and requirements.
        
        This method selects the best template and formats it appropriately.
        """
        
        # Determine best template based on patterns and focus
        if detected_patterns:
            if any(p in ["rfi", "submittal", "change_order"] for p in detected_patterns):
                template_name = "detailed_construction"
            elif "email" in detected_patterns:
                template_name = "few_shot_email" if use_examples else "concise_general"
            elif any(p in ["invoice", "po", "quote"] for p in detected_patterns):
                template_name = "cot_financial"
            else:
                template_name = "structured_mixed"
        else:
            # No patterns detected, use general approach
            template_name = "concise_general"
            
        return cls.get_prompt(
            template_name,
            prev_text,
            curr_text,
            patterns=detected_patterns
        )
    
    @staticmethod
    def _truncate_text(text: str, max_length: int) -> str:
        """Intelligently truncate text to fit token limits."""
        
        if not text:
            return "[EMPTY]"
            
        if len(text) <= max_length:
            return text
            
        # Try to break at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        break_point = max(last_period, last_newline)
        if break_point > max_length * 0.7:  # If we found a good break point
            return truncated[:break_point + 1] + "..."
        else:
            # Break at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."


# Specialized prompts for specific scenarios

def get_construction_specialist_prompt(
    prev_text: str,
    curr_text: str,
    project_context: Optional[str] = None
) -> str:
    """Specialized prompt for construction industry documents."""
    
    context_line = f"Project: {project_context}\n" if project_context else ""
    
    return f"""Construction document boundary analysis expert mode.
{context_line}
DOCUMENT TYPES: RFI, Submittal, Change Order, Daily Report, Meeting Minutes, Safety Report

PREVIOUS PAGE:
{prev_text[:500]}

CURRENT PAGE:
{curr_text[:500]}

Analyze for:
1. Document ID changes (RFI#, CO#, Submittal#)
2. Form headers and transmittals
3. Date/project reference changes
4. Signature blocks indicating document end

{{"boundary": true/false, "confidence": 0-1, "type": "rfi|submittal|co|daily|minutes|safety|other", "reason": "specific finding"}}"""


def get_email_chain_prompt(prev_text: str, curr_text: str) -> str:
    """Specialized prompt for email chain detection."""
    
    return f"""Email chain boundary detection:

PREV: {prev_text[-300:]}
CURR: {curr_text[:300]}

Rules:
- New "From:" after signature = new email
- "---- Original Message ----" = same chain
- Different subject = likely new email
- Forwarded/Reply headers = same chain

{{"boundary": bool, "confidence": 0-1, "type": "email|email_chain", "reason": str}}"""


def get_financial_document_prompt(
    prev_text: str,
    curr_text: str,
    include_patterns: bool = True
) -> str:
    """Specialized prompt for financial documents."""
    
    patterns_section = """
Key patterns:
- Invoice/PO/Quote numbers
- "Total:", "Amount Due:", "Balance:"
- Company names and addresses
- Date fields
- Payment terms
""" if include_patterns else ""
    
    return f"""Financial document boundary detection:
{patterns_section}
END OF PREVIOUS:
{prev_text[-400:]}

START OF CURRENT:
{curr_text[:400]}

{{"boundary": bool, "confidence": 0-1, "type": "invoice|po|quote|statement|receipt|other", "reason": str}}"""


def get_mixed_document_prompt(
    prev_text: str,
    curr_text: str,
    confidence_threshold: float = 0.7
) -> str:
    """General prompt for mixed document types."""
    
    return f"""Detect if current page starts a new document (confidence threshold: {confidence_threshold}).

Previous ends with:
{prev_text[-350:]}

Current starts with:
{curr_text[:350]}

Consider: headers, page numbers, context shifts, formatting changes.

{{"boundary": bool, "confidence": 0-1, "type": str, "reason": str (be specific)}}"""


# Prompt optimization utilities

def optimize_for_phi4_mini(prompt: str) -> str:
    """
    Optimize a prompt specifically for phi4-mini's characteristics.
    
    - Reduces redundancy
    - Ensures clear structure
    - Fits within token limits
    """
    
    # Remove excessive whitespace
    import re
    prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt)
    prompt = re.sub(r'  +', ' ', prompt)
    
    # Ensure prompt ends with clear instruction if JSON expected
    if "json" in prompt.lower() and not prompt.strip().endswith("}"):
        if not any(prompt.strip().endswith(x) for x in ["}", '"}', "}}'"]):
            prompt += "\n\nRespond only with valid JSON."
    
    # Add token limit warning if too long
    estimated_tokens = len(prompt) // 4  # Rough estimate
    if estimated_tokens > 3500:
        prompt = prompt[:14000] + "\n[Truncated for token limit]"
        
    return prompt.strip()


def create_adaptive_prompt(
    prev_text: str,
    curr_text: str,
    historical_accuracy: Optional[Dict[str, float]] = None
) -> str:
    """
    Create an adaptive prompt based on historical performance.
    
    Args:
        prev_text: Previous page text
        curr_text: Current page text  
        historical_accuracy: Dict of template_name -> accuracy score
    """
    
    if historical_accuracy:
        # Sort templates by historical accuracy
        best_template = max(
            historical_accuracy.items(),
            key=lambda x: x[1]
        )[0]
        
        # Use best performing template
        return Phi4MiniPromptLibrary.get_prompt(
            best_template,
            prev_text,
            curr_text
        )
    else:
        # Default to optimized selection
        return Phi4MiniPromptLibrary.get_optimized_prompt(
            prev_text,
            curr_text
        )


# Example usage
if __name__ == "__main__":
    # Example texts
    prev = "Sincerely,\nJohn Smith\nProject Manager"
    curr = "From: alice@example.com\nTo: bob@example.com\nSubject: RFI Response"
    
    # Get different prompt styles
    prompts = {
        "concise": Phi4MiniPromptLibrary.get_prompt("concise_general", prev, curr),
        "construction": get_construction_specialist_prompt(prev, curr),
        "email": get_email_chain_prompt(prev, curr),
        "optimized": Phi4MiniPromptLibrary.get_optimized_prompt(
            prev, curr, detected_patterns=["email", "rfi"]
        )
    }
    
    for name, prompt in prompts.items():
        print(f"\n=== {name.upper()} PROMPT ===")
        print(prompt)
        print(f"Estimated tokens: {len(prompt) // 4}")