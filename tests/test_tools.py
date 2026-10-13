import pytest
from tools import (
    get_weather,
    get_latest_news,
    get_directions_and_route,
    search_jobs,
    calculator,
    tools
)
from model_router import resolve_realtime_tools


def test_tool_registry():
    tool_names = [t.name for t in tools]
    assert "get_weather" in tool_names
    assert "get_latest_news" in tool_names
    assert "get_directions_and_route" in tool_names
    assert "search_jobs" in tool_names
    assert "calculator" in tool_names
    assert "web_search" in tool_names


def test_calculator_tool():
    assert calculator.invoke({"expression": "2 + 2"}) == "4"
    assert calculator.invoke({"expression": "math.sqrt(144)"}) == "12.0"
    assert calculator.invoke({"expression": "round(math.pi, 2)"}) == "3.14"


def test_weather_tool_execution():
    res = get_weather.invoke({"location": "London"})
    assert isinstance(res, str)
    assert len(res) > 10
    # Must contain weather details or informative status
    assert "Weather" in res or "London" in res


def test_news_tool_execution():
    res = get_latest_news.invoke({"query": "AI"})
    assert isinstance(res, str)
    assert len(res) > 10
    assert "News" in res or "Article" in res or "articles" in res


def test_directions_tool_execution():
    res = get_directions_and_route.invoke({
        "start_location": "Berlin",
        "end_location": "Munich",
        "mode": "driving-car"
    })
    assert isinstance(res, str)
    assert len(res) > 10
    assert "Route" in res or "Distance" in res or "Berlin" in res


def test_jobs_tool_execution():
    res = search_jobs.invoke({"query": "Python Developer", "location": "New York"})
    assert isinstance(res, str)
    assert len(res) > 10


def test_resolve_realtime_tools():
    # Weather detection
    w_res = resolve_realtime_tools("What is the weather in Tokyo right now?")
    assert w_res is not None
    assert "Tokyo" in w_res or "Weather" in w_res

    # News detection
    n_res = resolve_realtime_tools("What are the latest news about Artificial Intelligence?")
    assert n_res is not None
    assert "News" in n_res

    # Directions detection
    d_res = resolve_realtime_tools("Give me directions from Paris to Lyon")
    assert d_res is not None
    assert "Route" in d_res or "Distance" in d_res

    # Calculator detection
    c_res = resolve_realtime_tools("calculate 15 * 12")
    assert c_res is not None
    assert "180" in c_res

    # Non-tool query should return None
    assert resolve_realtime_tools("Can you write a poem about the sea?") is None
