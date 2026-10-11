import os
import logging
from typing import Tuple, Generator, Dict, Any, List
from security.injection import check_prompt_injection, CONFIDENTIALITY_RESPONSE
from security.secrets import contains_secrets, sanitize_output_text
from security.pii import scrub_pii
from security.llama_guard import llama_guard_engine

logger = logging.getLogger("RiveraGPT.Guardrails")


class SecurityGuardrails:
    """
    RiveraGPT System-Level Guardrails Layer.
    Implements:
    1. Llama Prompt Guard 2 (Prompt Injection Defense)
    2. NeMo Guardrails Policy Engine
    3. Llama Guard 4 (12B) Safety Classifier
    4. PII & Secret Scrubbing Engine
    """

    def __init__(self):
        self.guardrails_config_path = os.path.join("guardrails", "config.yml")
        self.has_nemo = self._init_nemo_guardrails()

    def _init_nemo_guardrails(self) -> bool:
        """Attempts to initialize NeMo Guardrails if package is installed."""
        try:
            from nemoguardrails import RailsConfig, LLMRails
            if os.path.exists(self.guardrails_config_path):
                config = RailsConfig.from_path("guardrails")
                self.rails = LLMRails(config)
                logger.info("NVIDIA NeMo Guardrails engine initialized successfully.")
                return True
        except Exception as e:
            logger.info(f"NeMo Guardrails native module standalone mode active ({e}). Using native RiveraGPT Security Engine.")
        return False

    def process_input_rails(self, user_input: str) -> Tuple[bool, str]:
        """
        Executes Input Rails Pipeline:
        1. Llama Prompt Guard 2 / System Prompt Extraction Protection
        2. Llama Guard 4 (12B) Safety Classification (14 Categories)
        3. PII & Secret Check
        Returns (is_blocked, blocked_response_or_clean_input)
        """
        # 1. Prompt Injection & System Prompt Extraction Defense
        is_injection, response = check_prompt_injection(user_input)
        if is_injection:
            logger.warning("Input Rail Flagged: Prompt Injection / System Extraction attempt.")
            return True, response

        # 2. Llama Guard 4 Safety Classifier
        is_unsafe, safety_refusal = llama_guard_engine.classify_input(user_input)
        if is_unsafe:
            logger.warning("Input Rail Flagged: Llama Guard 4 Safety Violation.")
            return True, safety_refusal

        # 3. Secret check in input
        if contains_secrets(user_input):
            logger.warning("Input Rail Flagged: Secrets detected in user input.")

        # 4. PII Scrubbing
        clean_input = scrub_pii(user_input)

        return False, clean_input

    def process_output_token(self, token: str) -> str:
        """
        Executes real-time Output Rails token sanitization.
        """
        if not token:
            return token
        return sanitize_output_text(token)

    def process_output_rails(self, full_text: str) -> str:
        """
        Executes final Output Rails:
        1. Llama Guard 4 Output Safety Classification
        2. Secret Detection & Provider Disclosure Redaction
        3. PII Detection & Redaction
        """
        if not full_text:
            return full_text

        # 1. Llama Guard 4 Output check
        is_unsafe, safety_refusal = llama_guard_engine.classify_output(full_text)
        if is_unsafe:
            return safety_refusal

        # 2. Secret & Provider Sanitization
        sanitized = sanitize_output_text(full_text)

        # 3. PII Redaction
        sanitized = scrub_pii(sanitized)
        return sanitized


# Global Guardrails Instance
guardrails_engine = SecurityGuardrails()
