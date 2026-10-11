import pytest
from model_router import detect_intent, get_ranked_fallback_chain, get_user_facing_mode_name
from model_registry import registry, normalize_parameters


def test_intent_detection():
    assert detect_intent("Help me debug this python function syntax") == "CODING"
    assert detect_intent("Prove by induction that 2^n > n") == "REASONING"
    assert detect_intent("नमस्ते, आप कैसे हैं?") == "INDIC_LANGUAGE"
    assert detect_intent("vanakkam, tell me a story") == "INDIC_LANGUAGE"
    assert detect_intent("What is the weather today?") == "CURRENT_INFO"
    assert detect_intent("Hello there!") == "SIMPLE_FAST"


def test_fallback_chain_generation():
    chain_fast = get_ranked_fallback_chain("fast", "SIMPLE_FAST")
    assert len(chain_fast) > 0
    assert isinstance(chain_fast, list)

    chain_reasoning = get_ranked_fallback_chain("reasoning", "REASONING")
    assert len(chain_reasoning) > 0

    chain_coding = get_ranked_fallback_chain("coding", "CODING")
    assert len(chain_coding) > 0


def test_mode_display_names():
    assert get_user_facing_mode_name("fast", "SIMPLE_FAST") == "RiveraGPT Fast"
    assert get_user_facing_mode_name("reasoning", "REASONING") == "RiveraGPT Reasoner"
    assert get_user_facing_mode_name("coding", "CODING") == "RiveraGPT Code"
    assert get_user_facing_mode_name("indic", "INDIC_LANGUAGE") == "RiveraGPT Indic"
    assert get_user_facing_mode_name("auto", "SIMPLE_FAST") == "RiveraGPT Fast"


def test_parameter_normalization():
    params_google = normalize_parameters("google", temperature=0.2, max_tokens=1000)
    assert params_google["temperature"] == 0.2
    assert params_google["max_output_tokens"] == 1000

    params_openai = normalize_parameters("openai", temperature=0.5, max_tokens=500)
    assert params_openai["temperature"] == 0.5
    assert params_openai["max_tokens"] == 500

    params_ollama = normalize_parameters("ollama", temperature=0.7, max_tokens=256)
    assert params_ollama["options"]["num_predict"] == 256


def test_registry_user_facing_modes():
    modes = registry.list_user_facing_modes()
    keys = [m["key"] for m in modes]
    assert "auto" in keys
    assert "fast" in keys
    assert "reasoning" in keys
    assert "coding" in keys
    assert "indic" in keys
