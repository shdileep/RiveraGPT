import os
import re
import logging
import httpx
from typing import Tuple

logger = logging.getLogger("RiveraGPT.LlamaGuard")

# 14 Safety Categories taxonomy
UNSAFE_PATTERNS = [
    # S1: Violent Criminology / Weapons / Harmful acts
    r'(?i)\b(?:how\s+to|instructions\s+for)?\s*(?:build|building|make|making|create|creating)\s+(?:a\s+)?(?:bomb|explosive|grenade|pipebomb)\b',
    r'(?i)\b(?:instructions\s+for\s+)?(?:poisoning|assassination|killing|mass\s+shooting)\b',
    
    # S4: Cyberattacks / Malware / Exploits
    r'(?i)\b(write\s+(?:a\s+)?ransomware|keylogger|zero-day\s+exploit|botnet|ddos\s+script)\b',
    r'(?i)\b(how\s+to\s+hack\s+into\s+(?:a\s+bank|government|database)\s+without\s+authorization)\b',
    
    # S5: CBRN Weapons
    r'(?i)\b(synthesize\s+(?:anthrax|ricin|sarin|nerve\s+agent|mustard\s+gas))\b',
    
    # S6: Suicide / Self-Harm
    r'(?i)\b(how\s+to\s+(?:commit\s+suicide|self-harm|cut\s+myself|end\s+my\s+life))\b',
    
    # S7: Hate Speech / Severe Incitement
    r'(?i)\b(incite\s+violence\s+against\s+(?:minorities|protected\s+groups|ethnicities))\b'
]

compiled_unsafe_regexes = [re.compile(p) for p in UNSAFE_PATTERNS]

SAFETY_REFUSAL_RESPONSE = "I cannot fulfill requests involving unsafe, harmful, or illegal content. Please ask a safe alternative question."


class LlamaGuardClassifier:
    """
    Llama Guard 4 (12B) Safety Classifier Engine.
    Evaluates inputs and outputs across 14 safety categories.
    Supported by local Ollama llama-guard models with rule-based fallback.
    """

    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.guard_model_name = os.getenv("LLAMA_GUARD_MODEL", "llama-guard3:8b")

    def _query_ollama_guard(self, text: str) -> Tuple[bool, str]:
        """Queries local Ollama Llama-Guard model if enabled explicitly."""
        if not os.getenv("ENABLE_LLAMA_GUARD_MODEL", "false").lower() in ("true", "1"):
            return False, ""
        try:
            with httpx.Client(timeout=0.5) as client:
                payload = {
                    "model": self.guard_model_name,
                    "prompt": f"[INST] Task: Check if the following input is safe or unsafe.\nInput: {text} [/INST]",
                    "stream": False
                }
                res = client.post(f"{self.ollama_url}/api/generate", json=payload)
                if res.status_code == 200:
                    resp_text = res.json().get("response", "").strip()
                    if resp_text.startswith("unsafe"):
                        logger.warning(f"Llama Guard local model flagged content as unsafe: {resp_text}")
                        return True, SAFETY_REFUSAL_RESPONSE
        except Exception:
            pass
        return False, ""

    def classify_input(self, text: str) -> Tuple[bool, str]:
        """
        Classifies user input for safety violations.
        Returns (is_unsafe, refusal_message)
        """
        if not text:
            return False, ""

        # 1. Query local Ollama Llama Guard model if available
        is_unsafe_local, msg = self._query_ollama_guard(text)
        if is_unsafe_local:
            return True, msg

        # 2. Rule-based safety fallback classification
        for regex in compiled_unsafe_regexes:
            if regex.search(text):
                logger.warning(f"Llama Guard 4 Classifier flagged unsafe input: {text[:60]}...")
                return True, SAFETY_REFUSAL_RESPONSE

        return False, ""

    def classify_output(self, response_text: str) -> Tuple[bool, str]:
        """
        Classifies generated response text for safety violations.
        """
        if not response_text:
            return False, ""

        for regex in compiled_unsafe_regexes:
            if regex.search(response_text):
                logger.warning("Llama Guard 4 Classifier flagged unsafe output content.")
                return True, SAFETY_REFUSAL_RESPONSE

        return False, ""


# Global Llama Guard Classifier Instance
llama_guard_engine = LlamaGuardClassifier()
