import re
from typing import Tuple

PROMPT_INJECTION_PATTERNS = [
    # Direct System Prompt Extraction
    r'(?i)\b(?:show|reveal|print|display|dump|get|output|return|share)\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules|developer\s+instructions|hidden\s+prompt|guidelines)\b',
    r'(?i)\bwhat\s+(?:are\s+)?your\s+(?:system\s+)?(?:instructions|prompt|rules|developer\s+instructions)\b',
    r'(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b',
    r'(?i)\bdump\s+(?:your\s+)?(?:configuration|config|env|environment|prompt)\b',
    r'(?i)\brepeat\s+(?:everything\s+)?(?:above|before|in\s+your\s+prompt)\b',
    
    # Indirect System Prompt Extraction (Encoding, Formatting, Acrostic, Translation, Roleplay)
    r'(?i)\b(?:encode|base64|hex|hexadecimal|json|yaml|markdown|xml)\s+(?:of\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules)\b',
    r'(?i)\btranslate\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules)\b',
    r'(?i)\b(?:acrostic|first\s+letter|first\s+word)\s+(?:of\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules)\b',
    r'(?i)\b(?:developer|debug|admin|root|DAN|jailbreak)\s+mode\b',
    r'(?i)\bpretend\s+(?:you\s+are|to\s+be)\s+(?:the\s+)?(?:developer|creator|admin|unconstrained)\b',
    r'(?i)\bhypothetical\s+scenario\s+where\s+you\s+(?:reveal|show|print)\b',

    # Model / Provider / Routing / Infrastructure / Tech Stack Probing (Direct & Indirect)
    r'(?i)\bwhat\s+(?:model|llm)\s+(?:are\s+you\s+running|do\s+you\s+use|is\s+this|are\s+you)\b',
    r'(?i)\bwhich\s+(?:api|provider|backend|engine)\s+(?:are\s+you\s+using|do\s+you\s+use|generated\s+this|powers\s+this)\b',
    r'(?i)\bwhich\s+model\s+(?:generated|produced|wrote)\s+this\b',
    r'(?i)\bwhat\s+is\s+your\s+(?:fallback\s+model|fallback\s+order|routing\s+logic|tech\s*stack|architecture|infrastructure)\b',
    r'(?i)\b(?:show|share|reveal|list|explain)\s+(?:your\s+)?(?:routing\s+logic|fallback\s+chain|infrastructure|tech\s*stack|architecture|frameworks|models|providers)\b',
    r'(?i)\bshow\s+(?:your\s+)?(?:environment\s+variables|api\s+keys|secrets|credentials)\b',
    r'(?i)\btell\s+me\s+what\'?s\s+behind\s+riveragpt\b',
    r'(?i)\bare\s+you\s+(?:powered\s+by|using|running\s+on)\s+(?:openai|gpt|gemini|claude|groq|qwen|llama|sarvam|smallest|ollama|deepseek|mistral)\b',
    r'(?i)\bwhat\s+provider\s+are\s+you\b',
    r'(?i)\breveal\s+the\s+underlying\s+model\b'
]

compiled_injection_regexes = [re.compile(p) for p in PROMPT_INJECTION_PATTERNS]

CONFIDENTIALITY_RESPONSE = "I’m RiveraGPT, an AI assistant. I can’t provide internal system or infrastructure details."


def check_prompt_injection(user_input: str) -> Tuple[bool, str]:
    """
    Scans input message for prompt extraction, system prompt probing, or injection.
    Returns (True, response) if flagged, (False, "") if safe.
    """
    if not user_input:
        return False, ""

    text = user_input.strip()

    for regex in compiled_injection_regexes:
        if regex.search(text):
            return True, CONFIDENTIALITY_RESPONSE

    return False, ""
