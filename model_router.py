import os
import time
import re
import logging
from typing import Dict, List, Any, Generator, Tuple
from model_registry import registry, health_tracker, normalize_parameters
from security.guardrails import guardrails_engine

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

logger = logging.getLogger("RiveraGPT.Router")


# Indic Script & Language Detection Regex
INDIC_SCRIPT_REGEX = re.compile(
    r'[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]'
)


def detect_intent(message: str) -> str:
    """
    Classifies user message intent into:
    - INDIC_LANGUAGE
    - REASONING
    - CODING
    - CURRENT_INFO
    - SIMPLE_FAST
    """
    text = message.strip()
    
    # 1. Indic Language Check
    if INDIC_SCRIPT_REGEX.search(text):
        return "INDIC_LANGUAGE"
    
    # Indic words written in Latin script
    indic_keywords = ["namaste", "vanakkam", "kya", "kaise", "telugu", "hindi", "tamil", "bengali", "marathi"]
    if any(k in text.lower() for k in indic_keywords):
        return "INDIC_LANGUAGE"

    # 2. Current Info Check
    current_keywords = ["today", "latest news", "current price", "weather", "release", "2026", "2025", "who won", "right now"]
    if any(k in text.lower() for k in current_keywords):
        return "CURRENT_INFO"

    # 3. Reasoning Check
    reasoning_keywords = ["prove", "theorem", "step by step", "algorithm", "derivative", "complexity", "architecture", "explain why"]
    if any(k in text.lower() for k in reasoning_keywords):
        return "REASONING"

    # 4. Coding Check
    coding_keywords = ["code", "python", "javascript", "function", "class", "bug", "syntax", "api", "html", "css", "sql", "error"]
    if any(k in text.lower() for k in coding_keywords):
        return "CODING"

    return "SIMPLE_FAST"


def get_ranked_fallback_chain(selected_model_key: str, intent: str) -> List[str]:
    """
    Builds a capability-aware fallback list of model keys.
    Supports user mode aliases ('fast', 'reasoning', 'coding', 'indic', 'auto').
    """
    all_models = registry.list_all_models()
    chain = []
    
    # Map user mode selection
    effective_intent = intent
    if selected_model_key == "fast":
        effective_intent = "SIMPLE_FAST"
    elif selected_model_key == "reasoning":
        effective_intent = "REASONING"
    elif selected_model_key == "coding":
        effective_intent = "CODING"
    elif selected_model_key == "indic":
        effective_intent = "INDIC_LANGUAGE"

    # Priority model allocation based on effective intent & available providers
    has_google = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    has_hf = bool(
        os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_API_KEY")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))

    if has_hf:
        if effective_intent == "CODING":
            priority_keys = [
                "huggingface/qwen-2.5-coder-32b",
                "huggingface/llama-3.3-70b",
                "huggingface/deepseek-r1-qwen-32b",
                "google/gemini-2.5-pro",
                "google/gemini-2.5-flash",
                "huggingface/gemma-r3",
                "huggingface/qwen-2.5-coder-7b",
                "huggingface/codellama-34b",
                "groq/llama-3.3-70b-versatile",
                "vllm/gemma-3-27b-it",
                "ollama/gemma:r3",
                "ollama/qwen3:30b",
                "openai/gpt-4o"
            ]
        elif effective_intent == "REASONING":
            priority_keys = [
                "huggingface/deepseek-r1-qwen-32b",
                "google/gemini-2.5-pro",
                "huggingface/llama-3.3-70b",
                "huggingface/qwen-2.5-72b",
                "google/gemini-2.5-flash",
                "huggingface/gemma-r3",
                "vllm/gemma-r3",
                "ollama/gemma:r3",
                "openai/gpt-4o",
                "groq/llama-3.3-70b-versatile",
                "ollama/deepseek-r1:7b"
            ]
        elif effective_intent == "INDIC_LANGUAGE":
            priority_keys = [
                "google/gemini-2.5-flash",
                "sarvam/sarvam-30b",
                "huggingface/qwen-2.5-72b",
                "huggingface/llama-3.3-70b",
                "sarvam/sarvam-105b",
                "openai/gpt-4o",
                "ollama/gemma:r3",
                "ollama/qwen3:8b"
            ]
        elif effective_intent == "SIMPLE_FAST":
            priority_keys = [
                "huggingface/qwen-2.5-coder-7b",
                "google/gemini-2.5-flash",
                "huggingface/llama-3.1-8b",
                "huggingface/gemma-2-9b",
                "groq/llama-3.1-8b-instant",
                "openai/gpt-4o-mini",
                "vllm/gemma-3-4b-it",
                "ollama/gemma:r3",
                "ollama/qwen3:8b"
            ]
        else:
            priority_keys = [
                "google/gemini-2.5-flash",
                "huggingface/qwen-2.5-coder-32b",
                "huggingface/llama-3.3-70b",
                "google/gemini-2.5-pro",
                "huggingface/gemma-r3",
                "huggingface/deepseek-r1-qwen-32b",
                "groq/llama-3.3-70b-versatile",
                "openai/gpt-4o-mini",
                "vllm/gemma-r3",
                "ollama/gemma:r3"
            ]
    elif has_google:
        if effective_intent == "INDIC_LANGUAGE":
            priority_keys = [
                "google/gemini-2.5-flash",
                "sarvam/sarvam-30b",
                "huggingface/qwen-2.5-72b",
                "openai/gpt-4o",
                "ollama/gemma:r3",
                "ollama/qwen3:8b"
            ]
        elif effective_intent == "REASONING":
            priority_keys = [
                "google/gemini-2.5-pro",
                "google/gemini-2.5-flash",
                "huggingface/deepseek-r1-qwen-32b",
                "huggingface/llama-3.3-70b",
                "huggingface/gemma-r3",
                "openai/gpt-4o",
                "ollama/gemma:r3",
                "vllm/gemma-r3",
                "groq/llama-3.3-70b-versatile"
            ]
        elif effective_intent == "CODING":
            priority_keys = [
                "google/gemini-2.5-pro",
                "google/gemini-2.5-flash",
                "huggingface/qwen-2.5-coder-32b",
                "huggingface/llama-3.3-70b",
                "huggingface/deepseek-r1-qwen-32b",
                "huggingface/gemma-r3",
                "huggingface/qwen-2.5-coder-7b",
                "groq/llama-3.3-70b-versatile",
                "openai/gpt-4o",
                "vllm/gemma-3-27b-it",
                "ollama/gemma:r3",
                "ollama/qwen3:30b"
            ]
        elif effective_intent == "SIMPLE_FAST":
            priority_keys = [
                "google/gemini-2.5-flash",
                "huggingface/qwen-2.5-coder-7b",
                "huggingface/llama-3.1-8b",
                "openai/gpt-4o-mini",
                "groq/llama-3.1-8b-instant",
                "ollama/gemma:r3",
                "vllm/gemma-3-4b-it",
                "ollama/qwen3:8b"
            ]
        else:
            priority_keys = [
                "google/gemini-2.5-flash",
                "google/gemini-2.5-pro",
                "huggingface/qwen-2.5-coder-32b",
                "huggingface/llama-3.3-70b",
                "huggingface/gemma-r3",
                "ollama/gemma:r3",
                "vllm/gemma-3-27b-it",
                "openai/gpt-4o-mini",
                "ollama/qwen3:8b"
            ]
    else:
        if effective_intent == "INDIC_LANGUAGE":
            priority_keys = ["sarvam/sarvam-30b", "huggingface/qwen-2.5-72b", "ollama/gemma:r3", "ollama/qwen3:8b", "openai/gpt-4o"]
        elif effective_intent == "REASONING":
            priority_keys = ["huggingface/deepseek-r1-qwen-32b", "huggingface/llama-3.3-70b", "huggingface/gemma-r3", "ollama/gemma:r3", "vllm/gemma-r3", "vllm/gemma-3-27b-it", "ollama/deepseek-r1:7b", "openai/gpt-4o", "groq/llama-3.3-70b-versatile"]
        elif effective_intent == "CODING":
            priority_keys = ["huggingface/qwen-2.5-coder-32b", "huggingface/llama-3.3-70b", "huggingface/deepseek-r1-qwen-32b", "huggingface/gemma-r3", "huggingface/qwen-2.5-coder-7b", "vllm/gemma-3-27b-it", "ollama/gemma:r3", "ollama/qwen3:30b", "groq/llama-3.3-70b-versatile", "openai/gpt-4o", "ollama/qwen3:8b"]
        elif effective_intent == "SIMPLE_FAST":
            priority_keys = ["huggingface/qwen-2.5-coder-7b", "huggingface/llama-3.1-8b", "ollama/gemma:r3", "vllm/gemma-3-4b-it", "ollama/qwen3:8b", "groq/llama-3.3-70b-versatile", "openai/gpt-4o-mini"]
        else:
            priority_keys = ["huggingface/qwen-2.5-coder-32b", "huggingface/llama-3.3-70b", "huggingface/gemma-r3", "ollama/gemma:r3", "vllm/gemma-3-27b-it", "ollama/qwen3:8b", "groq/llama-3.3-70b-versatile", "openai/gpt-4o-mini"]

    # Explicit backend key selected if provided
    if selected_model_key and selected_model_key not in ["auto", "fast", "reasoning", "coding", "indic"]:
        if registry.get_spec(selected_model_key):
            chain.append(selected_model_key)

    for k in priority_keys:
        if k not in chain and registry.get_spec(k):
            chain.append(k)

    # Append remaining catalog models
    for m in all_models:
        k = m["key"]
        if k not in chain:
            chain.append(k)

    return chain


def sanitize_message_alternation(system_prompt: str, messages: List[Any]) -> List[Any]:
    """
    Ensures message turns strictly alternate between User and Assistant,
    coalesces adjacent messages of identical roles, and places System prompt first.
    """
    sanitized = []
    for m in messages:
        c = str(getattr(m, "content", "") or "").strip()
        if not c:
            continue
        if not sanitized:
            if isinstance(m, AIMessage):
                continue
            sanitized.append(m)
        else:
            last = sanitized[-1]
            if type(last) is type(m):
                sanitized[-1] = type(last)(content=str(last.content) + "\n\n" + c)
            else:
                sanitized.append(m)
    return [SystemMessage(content=system_prompt)] + sanitized


def execute_single_provider_stream(
    model_key: str,
    messages: List[Any],
    system_prompt: str
) -> Generator[Dict[str, Any], None, None]:
    """
    Executes a single provider model call. Yields dict chunks with sanitized tokens.
    """
    spec = registry.get_spec(model_key)
    if not spec:
        raise ValueError(f"Unknown model spec")

    provider = spec["provider"]
    model_id = spec["model_id"]

    start_time = time.time()
    token_count = 0

    # Ensure system prompt is first message and turns alternate cleanly
    full_messages = sanitize_message_alternation(system_prompt, messages)

    # --- 1. Google Gemini Provider ---
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("Google API key is missing or not configured.")

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key,
            temperature=0.3,
            streaming=True
        )

        for chunk in llm.stream(full_messages):
            content = getattr(chunk, "content", "")
            if content:
                token_count += len(str(content).split())
                sanitized_token = guardrails_engine.process_output_token(str(content))
                yield {
                    "token": sanitized_token
                }

    # --- 2. OpenAI / Groq / Smallest AI / Sarvam / vLLM / Hugging Face OpenAI-Compatible Providers ---
    elif provider in ["openai", "groq", "smallest", "sarvam", "vllm", "huggingface"]:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None
        elif provider == "huggingface":
            api_key = (
                os.getenv("HUGGINGFACE_API_KEY")
                or os.getenv("HF_TOKEN")
                or os.getenv("HUGGING_FACE_API_KEY")
                or os.getenv("HUGGINGFACE_HUB_TOKEN")
            )
            base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            base_url = "https://api.groq.com/openai/v1"
        elif provider == "smallest":
            api_key = os.getenv("SMALLEST_AI_API_KEY")
            base_url = "https://api.smallest.ai/v1"
        elif provider == "sarvam":
            api_key = os.getenv("SARVAM_API_KEY")
            base_url = "https://api.sarvam.ai/v1"
        elif provider == "vllm":
            api_key = os.getenv("VLLM_API_KEY", "EMPTY")
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        else:
            api_key = None
            base_url = None

        if provider != "vllm" and (not api_key or api_key.startswith("your_")):
            raise ValueError(f"{provider.capitalize()} API key is missing or not configured.")

        success = False
        if ChatOpenAI is not None:
            try:
                kwargs = {
                    "model": model_id,
                    "api_key": api_key if api_key else "EMPTY",
                    "temperature": 0.3,
                    "streaming": True
                }
                if base_url:
                    kwargs["base_url"] = base_url

                llm = ChatOpenAI(**kwargs)
                for chunk in llm.stream(full_messages):
                    content = getattr(chunk, "content", "")
                    if content:
                        token_count += len(str(content).split())
                        sanitized_token = guardrails_engine.process_output_token(str(content))
                        yield {
                            "token": sanitized_token
                        }
                success = True
            except Exception as stream_err:
                if provider != "huggingface":
                    raise stream_err
                logger.warning(f"ChatOpenAI stream failed for {model_id}: {stream_err}. Attempting direct HTTP fallback...")

        if not success:
            import json
            import httpx
            endpoint = f"{base_url.rstrip('/')}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            formatted_msgs = []
            for m in full_messages:
                role = "user"
                if isinstance(m, SystemMessage):
                    role = "system"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                formatted_msgs.append({"role": role, "content": str(m.content)})

            body = {
                "model": model_id,
                "messages": formatted_msgs,
                "temperature": 0.3,
                "stream": True
            }
            with httpx.stream("POST", endpoint, headers=headers, json=body, timeout=45.0) as resp:
                if resp.status_code >= 400:
                    raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            item = json.loads(data_str)
                            delta = item.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                token_count += len(delta.split())
                                sanitized_token = guardrails_engine.process_output_token(delta)
                                yield {"token": sanitized_token}
                        except Exception:
                            continue

    # --- 3. Ollama Local Provider ---
    elif provider == "ollama":
        import ollama
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        formatted_messages = []
        for msg in full_messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            formatted_messages.append({"role": role, "content": msg.content})

        client = ollama.Client(host=ollama_base)
        res_stream = client.chat(model=model_id, messages=formatted_messages, stream=True)

        for chunk in res_stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                token_count += len(token.split())
                sanitized_token = guardrails_engine.process_output_token(token)
                yield {
                    "token": sanitized_token
                }

    else:
        raise ValueError("Unsupported provider execution")

    # Record metrics internally
    duration_ms = (time.time() - start_time) * 1000.0
    health_tracker.record_success(model_key, duration_ms, token_count)


def get_user_facing_mode_name(selected_key: str, intent: str) -> str:
    """Returns abstract RiveraGPT display mode name."""
    if selected_key == "fast" or intent == "SIMPLE_FAST":
        return "RiveraGPT Fast"
    elif selected_key == "reasoning" or intent == "REASONING":
        return "RiveraGPT Reasoner"
    elif selected_key == "coding" or intent == "CODING":
        return "RiveraGPT Code"
    elif selected_key == "indic" or intent == "INDIC_LANGUAGE":
        return "RiveraGPT Indic"
    return "RiveraGPT Auto"


def execute_with_capability_fallback(
    user_message: str,
    messages: List[Any],
    system_prompt: str,
    selected_model_key: str = "auto"
) -> Generator[Dict[str, Any], None, None]:
    """
    RiveraGPT Core Router Engine with Security Guardrails:
    1. Runs Input Security, PII, and Prompt Injection Guardrails
    2. Classifies intent & builds capability fallback chain
    3. Executes execution loop with stream output sanitization
    """
    # --- Input Security Guardrails ---
    is_blocked, guardrail_result = guardrails_engine.process_input_rails(user_message)
    if is_blocked:
        yield {
            "meta": {
                "display_name": "RiveraGPT Shield",
                "mode": "security"
            }
        }
        yield {"token": guardrail_result}
        return

    # Use scrubbed input message if clean
    clean_user_message = guardrail_result if guardrail_result else user_message

    intent = detect_intent(clean_user_message)
    fallback_chain = get_ranked_fallback_chain(selected_model_key, intent)

    mode_display = get_user_facing_mode_name(selected_model_key, intent)

    logger.info(f"RiveraGPT Request | Intent: '{intent}' | Mode: '{mode_display}'")

    for model_key in fallback_chain:
        spec = registry.get_spec(model_key)
        if not spec:
            continue

        health = health_tracker.get_health(model_key)
        if health.get("status") == "UNAVAILABLE":
            continue

        try:
            # Internal logging only
            logger.info(f"Executing internally via provider '{spec['provider']}' model '{spec['model_id']}'")
            
            first_chunk = True
            for chunk in execute_single_provider_stream(model_key, messages, system_prompt):
                if first_chunk:
                    # Send abstract telemetry event (no provider, model, or intent disclosure)
                    yield {
                        "meta": {
                            "display_name": mode_display,
                            "mode": selected_model_key if selected_model_key != "auto" else "auto"
                        }
                    }
                    first_chunk = False
                yield chunk
            
            # Successful execution complete
            return

        except Exception as e:
            error_str = str(e)
            logger.warning(f"Internal provider failure on '{model_key}': {error_str}. Attempting fallback...")
            
            status_code = 429 if ("429" in error_str or "rate limit" in error_str.lower()) else 500
            health_tracker.record_error(model_key, error_str, status_code)
            continue

    # Generic user-facing error response
    yield {
        "error": "Sorry, I couldn't complete that request. Please try again."
    }

