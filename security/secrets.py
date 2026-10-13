import re
from typing import List

# Patterns for sensitive keys, credentials, tokens, paths, IPs, and internal provider models
SECRET_PATTERNS = [
    # API Keys / Tokens
    r'(?i)(api[_-]?key|secret[_-]?key|auth[_-]?token|access[_-]?token|bearer[_-]?token)\s*[:=]\s*["\']?[A-Za-z0-9_\-\.]{8,}["\']?',
    r'sk-[A-Za-z0-9]{20,}',
    r'gsk_[A-Za-z0-9]{20,}',
    r'AIzaSy[A-Za-z0-9_\-]{33}',
    r'hf_[A-Za-z0-9]{30,}',
    r'tvly-[A-Za-z0-9_\-]{20,}',
    r'AQ\.Ab8[A-Za-z0-9_\-]{30,}',
    r'eyJ[A-Za-z0-9_\-\.]{30,}={0,2}',
    
    # Environment variables & DB connection strings
    r'(?i)(mongodb(\+srv)?|postgres|postgresql|mysql|redis):\/\/[^\s]+',
    r'(?i)[A-Z0-9_]+=(?:[A-Za-z0-9_\-\.\/]{8,})',
    
    # Private IP addresses & internal file paths
    r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b',
    r'(?:[a-zA-Z]:\\(?:[^\s\\]+\\)+[^\s\\]+)',
    r'(?:\/(?:[^\s\/]+\/)+[^\s\/]+)'
]

INTERNAL_PROVIDER_MODEL_PATTERNS = [
    r'(?i)\b(groq|openai|gemini|claude|llama|qwen|sarvam|smallest\s*ai|ollama|nvidia)\b',
    r'(?i)\b(llama-3\.[13]-(?:8b|70b)|gpt-4o|gemini-2\.5-(?:flash|pro)|sarvam-30b|sarvam-105b|qwen3:8b)\b'
]

compiled_secret_regexes = [re.compile(p) for p in SECRET_PATTERNS]
compiled_provider_regexes = [re.compile(p) for p in INTERNAL_PROVIDER_MODEL_PATTERNS]


def contains_secrets(text: str) -> bool:
    """Checks if text contains secret keys, credentials, or internal configuration."""
    if not text:
        return False
    for regex in compiled_secret_regexes:
        if regex.search(text):
            return True
    return False


def sanitize_output_text(text: str) -> str:
    """
    Sanitizes assistant response text:
    - Redacts any leaked secrets or credentials
    - Removes accidental disclosure of backend provider/model names
    """
    if not text:
        return text

    sanitized = text

    # Redact secrets
    for regex in compiled_secret_regexes:
        sanitized = regex.sub("[CONFIDENTIAL_REDACTED]", sanitized)

    # Redact explicit backend provider / model disclosures if present
    for regex in compiled_provider_regexes:
        sanitized = regex.sub("RiveraGPT", sanitized)

    # Redact accidental leaks of model orchestration, router, or provider lists
    architecture_patterns = [
        r'(?i)\bmodel\s+orchestration\s+system\b',
        r'(?i)\bmodel\s+router\b',
        r'(?i)\bautomatic\s+model\s+routing\b',
        r'(?i)\bmulti-model\s+orchestrator\b',
        r'(?i)\bproviding\s+automatic\s+model\s+routing[^\.\n]*',
        r'(?i)\bacross\s+(?:OpenAI|Google|Groq|Sarvam|Smallest|Ollama)[^\.\n]*'
    ]
    for p in architecture_patterns:
        sanitized = re.sub(p, "AI assistant", sanitized)

    # Clean excessive random special character noise (e.g. --- +++"""""#####%!@#$())
    sanitized = re.sub(r'[\+\=%!@\$]{4,}', '', sanitized)
    sanitized = re.sub(r'"{4,}', '"', sanitized)
    sanitized = re.sub(r"'{4,}", "'", sanitized)
    sanitized = re.sub(r'#{6,}', '###', sanitized)

    return sanitized
