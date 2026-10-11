import re

PII_PATTERNS = [
    # Emails
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL_REDACTED]'),
    # Phone numbers
    (r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE_REDACTED]'),
    # SSN
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
    # Credit Card
    (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CARD_REDACTED]')
]

compiled_pii_patterns = [(re.compile(pattern), replacement) for pattern, replacement in PII_PATTERNS]


def scrub_pii(text: str) -> str:
    """Scrubs personal identifiable information (PII) from text."""
    if not text:
        return text
    result = text
    for regex, replacement in compiled_pii_patterns:
        result = regex.sub(replacement, result)
    return result
