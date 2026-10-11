import os
import time
import logging
from typing import Dict, List, Any, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("RiveraGPT.Registry")
logging.basicConfig(level=logging.INFO)

# Default Model Specifications & Capabilities
STATIC_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    # Groq Models (Very Fast)
    "groq/llama-3.1-8b-instant": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "display_name": "Groq Llama 3.1 8B Instant (Ultra-Fast)",
        "capabilities": ["chat", "fast", "coding"],
        "context_window": 131072,
        "speed_score": 98,
        "quality_score": 82,
        "is_local": False,
        "cost_tier": "very_low"
    },
    "groq/llama-3.3-70b-versatile": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "display_name": "Groq Llama 3.3 70B Versatile",
        "capabilities": ["chat", "reasoning", "coding"],
        "context_window": 128000,
        "speed_score": 90,
        "quality_score": 92,
        "is_local": False,
        "cost_tier": "low"
    },
    
    # Gemini Models (Multimodal, Long Context)
    "google/gemini-2.5-flash": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash",
        "capabilities": ["chat", "fast", "vision", "reasoning", "coding"],
        "context_window": 1048576,
        "speed_score": 92,
        "quality_score": 90,
        "is_local": False,
        "cost_tier": "low"
    },
    "google/gemini-2.5-pro": {
        "provider": "google",
        "model_id": "gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro (Deep Reasoning)",
        "capabilities": ["chat", "reasoning", "vision", "coding"],
        "context_window": 2097152,
        "speed_score": 78,
        "quality_score": 96,
        "is_local": False,
        "cost_tier": "medium"
    },
    "google/gemini-2.5-flash-lite": {
        "provider": "google",
        "model_id": "gemini-2.5-flash-lite",
        "display_name": "Gemini 2.5 Flash Lite",
        "capabilities": ["chat", "fast"],
        "context_window": 1048576,
        "speed_score": 95,
        "quality_score": 85,
        "is_local": False,
        "cost_tier": "very_low"
    },

    # OpenAI Models (Flagship & Reasoning)
    "openai/gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "display_name": "OpenAI GPT-4o",
        "capabilities": ["chat", "reasoning", "vision", "coding"],
        "context_window": 128000,
        "speed_score": 85,
        "quality_score": 95,
        "is_local": False,
        "cost_tier": "high"
    },
    "openai/gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "display_name": "OpenAI GPT-4o Mini",
        "capabilities": ["chat", "fast", "coding"],
        "context_window": 128000,
        "speed_score": 93,
        "quality_score": 88,
        "is_local": False,
        "cost_tier": "low"
    },

    # Sarvam AI Models (Indic Languages & Speech)
    "sarvam/sarvam-30b": {
        "provider": "sarvam",
        "model_id": "sarvam-30b",
        "display_name": "Sarvam 30B (Indic LLM)",
        "capabilities": ["chat", "indic_language"],
        "context_window": 65536,
        "speed_score": 80,
        "quality_score": 89,
        "is_local": False,
        "cost_tier": "medium"
    },
    "sarvam/sarvam-105b": {
        "provider": "sarvam",
        "model_id": "sarvam-105b",
        "display_name": "Sarvam 105B (Deep Indic Reasoning)",
        "capabilities": ["chat", "indic_language", "reasoning"],
        "context_window": 131072,
        "speed_score": 72,
        "quality_score": 94,
        "is_local": False,
        "cost_tier": "medium"
    },

    # Smallest AI (Electron LLM)
    "smallest/electron": {
        "provider": "smallest",
        "model_id": "electron",
        "display_name": "Smallest AI Electron",
        "capabilities": ["chat", "fast", "voice_stack"],
        "context_window": 32768,
        "speed_score": 94,
        "quality_score": 86,
        "is_local": False,
        "cost_tier": "low"
    },

    # Local Ollama Models (Qwen3, Gemma3, DeepSeek-R1, Phi-4, Llama 3)
    "ollama/qwen3:8b": {
        "provider": "ollama",
        "model_id": "qwen3:8b",
        "display_name": "Local Qwen3 8B (Everyday General & Coding)",
        "capabilities": ["chat", "fast", "coding", "reasoning"],
        "context_window": 32768,
        "speed_score": 92,
        "quality_score": 88,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/qwen3:14b": {
        "provider": "ollama",
        "model_id": "qwen3:14b",
        "display_name": "Local Qwen3 14B (Quality Reasoning & Coding)",
        "capabilities": ["chat", "reasoning", "coding"],
        "context_window": 32768,
        "speed_score": 85,
        "quality_score": 92,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/qwen3:30b": {
        "provider": "ollama",
        "model_id": "qwen3:30b",
        "display_name": "Local Qwen3 30B (Powerful MoE Reasoning)",
        "capabilities": ["chat", "reasoning", "coding"],
        "context_window": 32768,
        "speed_score": 80,
        "quality_score": 95,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/gemma3:4b": {
        "provider": "ollama",
        "model_id": "gemma3:4b",
        "display_name": "Local Gemma 3 4B (Fast General & Vision)",
        "capabilities": ["chat", "fast", "vision"],
        "context_window": 16384,
        "speed_score": 95,
        "quality_score": 84,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/gemma3:12b": {
        "provider": "ollama",
        "model_id": "gemma3:12b",
        "display_name": "Local Gemma 3 12B (General & Vision)",
        "capabilities": ["chat", "vision", "reasoning"],
        "context_window": 32768,
        "speed_score": 88,
        "quality_score": 90,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/gemma3:27b": {
        "provider": "ollama",
        "model_id": "gemma3:27b",
        "display_name": "Local Gemma 3 27B (High Quality General & Vision)",
        "capabilities": ["chat", "vision", "reasoning"],
        "context_window": 32768,
        "speed_score": 82,
        "quality_score": 94,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/gemma:r3": {
        "provider": "ollama",
        "model_id": "gemma:r3",
        "display_name": "Local Gemma R3 (Fast General & Reasoning)",
        "capabilities": ["chat", "fast", "vision", "reasoning", "coding"],
        "context_window": 32768,
        "speed_score": 93,
        "quality_score": 91,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/gemma:7b": {
        "provider": "ollama",
        "model_id": "gemma:7b",
        "display_name": "Local Gemma 7B",
        "capabilities": ["chat", "fast"],
        "context_window": 8192,
        "speed_score": 90,
        "quality_score": 86,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/deepseek-r1:7b": {
        "provider": "ollama",
        "model_id": "deepseek-r1:7b",
        "display_name": "Local DeepSeek-R1 7B (Reasoning Specialist)",
        "capabilities": ["chat", "reasoning"],
        "context_window": 16384,
        "speed_score": 86,
        "quality_score": 93,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/deepseek-r1:14b": {
        "provider": "ollama",
        "model_id": "deepseek-r1:14b",
        "display_name": "Local DeepSeek-R1 14B (Deep Reasoning)",
        "capabilities": ["chat", "reasoning"],
        "context_window": 32768,
        "speed_score": 80,
        "quality_score": 96,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/phi4": {
        "provider": "ollama",
        "model_id": "phi4",
        "display_name": "Local Phi-4 14B (Reasoning & Code)",
        "capabilities": ["chat", "reasoning", "coding"],
        "context_window": 16384,
        "speed_score": 84,
        "quality_score": 91,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/mistral:7b": {
        "provider": "ollama",
        "model_id": "mistral:7b",
        "display_name": "Local Mistral 7B (Lightweight Fallback)",
        "capabilities": ["chat", "fast", "coding"],
        "context_window": 8192,
        "speed_score": 91,
        "quality_score": 85,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/llama3.1:8b": {
        "provider": "ollama",
        "model_id": "llama3.1:8b",
        "display_name": "Local Llama 3.1 8B (Stable Fallback)",
        "capabilities": ["chat", "fast", "coding"],
        "context_window": 131072,
        "speed_score": 90,
        "quality_score": 86,
        "is_local": True,
        "cost_tier": "free"
    },
    "ollama/llama3.3:70b": {
        "provider": "ollama",
        "model_id": "llama3.3:70b",
        "display_name": "Local Llama 3.3 70B (Powerful Local)",
        "capabilities": ["chat", "reasoning", "coding"],
        "context_window": 128000,
        "speed_score": 75,
        "quality_score": 95,
        "is_local": True,
        "cost_tier": "free"
    },

    # High-Performance vLLM Server Models (Gemma R3, Gemma 3, Llama, Qwen)
    "vllm/gemma-3-27b-it": {
        "provider": "vllm",
        "model_id": "gemma-3-27b-it",
        "display_name": "vLLM Gemma 3 27B IT (High-Throughput)",
        "capabilities": ["chat", "vision", "reasoning", "coding"],
        "context_window": 32768,
        "speed_score": 94,
        "quality_score": 95,
        "is_local": True,
        "cost_tier": "free"
    },
    "vllm/gemma-3-12b-it": {
        "provider": "vllm",
        "model_id": "gemma-3-12b-it",
        "display_name": "vLLM Gemma 3 12B IT",
        "capabilities": ["chat", "vision", "reasoning", "fast"],
        "context_window": 32768,
        "speed_score": 96,
        "quality_score": 91,
        "is_local": True,
        "cost_tier": "free"
    },
    "vllm/gemma-3-4b-it": {
        "provider": "vllm",
        "model_id": "gemma-3-4b-it",
        "display_name": "vLLM Gemma 3 4B IT (Ultra-Fast)",
        "capabilities": ["chat", "fast", "vision"],
        "context_window": 16384,
        "speed_score": 98,
        "quality_score": 86,
        "is_local": True,
        "cost_tier": "free"
    },
    "vllm/gemma-r3": {
        "provider": "vllm",
        "model_id": "gemma-r3",
        "display_name": "vLLM Gemma R3",
        "capabilities": ["chat", "reasoning", "coding", "fast", "vision"],
        "context_window": 32768,
        "speed_score": 95,
        "quality_score": 92,
        "is_local": True,
        "cost_tier": "free"
    }
}


class ModelHealthTracker:
    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        for key in STATIC_MODEL_CATALOG:
            self._init_key(key)

    def _init_key(self, key: str):
        if key not in self.metrics:
            self.metrics[key] = {
                "status": "HEALTHY", # HEALTHY, DEGRADED, UNAVAILABLE
                "latency_ms": 150.0,
                "ttft_ms": 80.0,
                "tokens_per_sec": 85.0,
                "success_count": 10,
                "error_count": 0,
                "429_count": 0,
                "last_success": time.time(),
                "last_error": None,
                "error_msg": None
            }

    def record_success(self, key: str, latency_ms: float, tokens_count: int = 50):
        self._init_key(key)
        m = self.metrics[key]
        m["success_count"] += 1
        # Exponential moving average for latency
        m["latency_ms"] = round(0.7 * m["latency_ms"] + 0.3 * latency_ms, 1)
        if latency_ms > 0 and tokens_count > 0:
            tps = (tokens_count / (latency_ms / 1000.0))
            m["tokens_per_sec"] = round(0.7 * m["tokens_per_sec"] + 0.3 * tps, 1)
        m["last_success"] = time.time()
        m["status"] = "HEALTHY"

    def record_error(self, key: str, error_msg: str, status_code: int = 500):
        self._init_key(key)
        m = self.metrics[key]
        m["error_count"] += 1
        if status_code == 429 or "google/" in key:
            if status_code == 429:
                m["429_count"] += 1
            m["status"] = "DEGRADED"
        else:
            m["status"] = "UNAVAILABLE"
        m["last_error"] = time.time()
        m["error_msg"] = str(error_msg)[:200]

    def get_health(self, key: str) -> Dict[str, Any]:
        self._init_key(key)
        return self.metrics[key]


health_tracker = ModelHealthTracker()


class RiveraModelRegistry:
    def __init__(self):
        self.catalog = dict(STATIC_MODEL_CATALOG)
        self.last_refresh_time = 0.0


    def refresh_catalog(self, force: bool = False):
        """Discover live available models from Groq, Ollama, and vLLM if reachable (60s TTL cache)."""
        now = time.time()
        if not force and (now - self.last_refresh_time < 60.0):
            return

        self.last_refresh_time = now
        self._discover_groq()
        ollama_online = self._discover_ollama()
        vllm_online = self._discover_vllm()

        if not ollama_online:
            for key, spec in self.catalog.items():
                if spec.get("provider") == "ollama":
                    health_tracker.record_error(key, "Ollama service offline", 503)
        if not vllm_online:
            for key, spec in self.catalog.items():
                if spec.get("provider") == "vllm":
                    health_tracker.record_error(key, "vLLM service offline", 503)

    def _discover_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key.startswith("your_"):
            return
        try:
            with httpx.Client(timeout=0.5) as client:
                res = client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                if res.status_code == 200:
                    data = res.json()
                    for m_obj in data.get("data", []):
                        m_id = m_obj.get("id")
                        key = f"groq/{m_id}"
                        if key not in self.catalog:
                            self.catalog[key] = {
                                "provider": "groq",
                                "model_id": m_id,
                                "display_name": f"Groq {m_id}",
                                "capabilities": ["chat", "fast"],
                                "context_window": 8192,
                                "speed_score": 95,
                                "quality_score": 85,
                                "is_local": False,
                                "cost_tier": "low"
                            }
        except Exception as e:
            logger.debug(f"Groq auto-discovery skipped: {e}")

    def _discover_ollama(self) -> bool:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            with httpx.Client(timeout=0.5) as client:
                res = client.get(f"{ollama_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    for m_obj in data.get("models", []):
                        m_name = m_obj.get("name")
                        key = f"ollama/{m_name}"
                        if key not in self.catalog:
                            caps = ["chat"]
                            m_lower = m_name.lower()
                            if "deepseek" in m_lower or "reason" in m_lower or "r1" in m_lower:
                                caps.extend(["reasoning"])
                            if "qwen" in m_lower or "code" in m_lower or "phi" in m_lower:
                                caps.extend(["coding", "reasoning", "fast"])
                            if "gemma" in m_lower:
                                caps.extend(["vision", "fast", "reasoning", "coding"])

                            self.catalog[key] = {
                                "provider": "ollama",
                                "model_id": m_name,
                                "display_name": f"Local Ollama ({m_name})",
                                "capabilities": list(set(caps)),
                                "context_window": 16384,
                                "speed_score": 88,
                                "quality_score": 88,
                                "is_local": True,
                                "cost_tier": "free"
                            }
                    return True
        except Exception as e:
            logger.debug(f"Ollama auto-discovery skipped: {e}")
        return False

    def _discover_vllm(self) -> bool:
        vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        headers = {}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            with httpx.Client(timeout=0.5) as client:
                res = client.get(f"{vllm_url.rstrip('/')}/models", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    for m_obj in data.get("data", []):
                        m_id = m_obj.get("id")
                        if not m_id:
                            continue
                        key = f"vllm/{m_id}"
                        if key not in self.catalog:
                            caps = ["chat"]
                            m_lower = m_id.lower()
                            if "gemma" in m_lower:
                                caps.extend(["vision", "fast", "reasoning", "coding"])
                            if "code" in m_lower or "qwen" in m_lower or "coder" in m_lower:
                                caps.extend(["coding", "reasoning"])
                            if "r1" in m_lower or "reason" in m_lower:
                                caps.extend(["reasoning"])

                            self.catalog[key] = {
                                "provider": "vllm",
                                "model_id": m_id,
                                "display_name": f"vLLM Server ({m_id})",
                                "capabilities": list(set(caps)),
                                "context_window": 32768,
                                "speed_score": 95,
                                "quality_score": 90,
                                "is_local": True,
                                "cost_tier": "free"
                            }
                    return True
        except Exception as e:
            logger.debug(f"vLLM auto-discovery skipped: {e}")
        return False

    def list_all_models(self) -> List[Dict[str, Any]]:
        self.refresh_catalog()
        result = []
        for key, spec in self.catalog.items():
            h = health_tracker.get_health(key)
            item = dict(spec)
            item["key"] = key
            item["health"] = h["status"]
            item["latency_ms"] = h["latency_ms"]
            item["tokens_per_sec"] = h["tokens_per_sec"]
            result.append(item)
        return result

    def list_user_facing_modes(self) -> List[Dict[str, Any]]:
        """Returns public RiveraGPT modes without exposing provider/model details."""
        return [
            {"key": "auto", "display_name": "⚡ RiveraGPT Auto", "description": "Auto-Router (Fastest + Smartest)"},
            {"key": "fast", "display_name": "⚡ RiveraGPT Fast", "description": "Optimized for speed"},
            {"key": "reasoning", "display_name": "🧠 RiveraGPT Reasoner", "description": "Deep reasoning & analysis"},
            {"key": "coding", "display_name": "💻 RiveraGPT Code", "description": "Code generation & debugging"},
            {"key": "indic", "display_name": "🇮🇳 RiveraGPT Indic", "description": "Indian languages & translation"}
        ]

    def get_spec(self, key: str) -> Optional[Dict[str, Any]]:
        return self.catalog.get(key)


registry = RiveraModelRegistry()


def normalize_parameters(
    provider: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    stream: bool = True
) -> Dict[str, Any]:
    """
    Standard Rivera parameters normalized into provider-specific format.
    """
    if provider == "google":
        return {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
    elif provider in ["openai", "groq", "smallest", "sarvam", "vllm"]:
        return {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream
        }
    elif provider == "ollama":
        return {
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
