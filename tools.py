import os
import math
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv
from langchain_core.tools import tool
from database import save_memory, search_memory
from rag import retrieve_from_rag

load_dotenv()

CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


# ==============================================================================
# 1. LIVE WEATHER TOOL (Dual-Provider: WeatherAPI + OpenWeatherMap Fallback)
# ==============================================================================
@tool
def get_weather(location: str) -> str:
    """
    Get real-time weather conditions and forecast for any city or location worldwide.
    Provides temperature (both Celsius and Fahrenheit), weather conditions, humidity,
    wind speed/direction, and today's forecast.
    Example: get_weather("London") or get_weather("Tokyo, Japan")
    """
    cleaned_loc = location.strip()
    if not cleaned_loc:
        return "Please specify a city or location to get the weather for."

    weatherapi_key = os.getenv("WEATHERAPI_KEY") or os.getenv("WEATHER_API_KEY")
    owm_key = os.getenv("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY")

    # 1. Try WeatherAPI.com (Primary)
    if weatherapi_key and not weatherapi_key.startswith("your_"):
        try:
            encoded_loc = urllib.parse.quote(cleaned_loc)
            url = f"https://api.weatherapi.com/v1/forecast.json?key={weatherapi_key}&q={encoded_loc}&days=1&aqi=no"
            req = urllib.request.Request(url, headers={"User-Agent": "RiveraGPT/2.0"})
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                loc_info = data.get("location", {})
                curr = data.get("current", {})
                forecast_day = data.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})

                place = f"{loc_info.get('name', cleaned_loc)}"
                region = loc_info.get("region", "")
                country = loc_info.get("country", "")
                full_place = ", ".join([p for p in [place, region, country] if p])

                temp_c = curr.get("temp_c", "N/A")
                temp_f = curr.get("temp_f", "N/A")
                feels_c = curr.get("feelslike_c", "N/A")
                feels_f = curr.get("feelslike_f", "N/A")
                condition = curr.get("condition", {}).get("text", "Unknown")
                humidity = curr.get("humidity", "N/A")
                wind_kph = curr.get("wind_kph", "N/A")
                wind_dir = curr.get("wind_dir", "")
                uv = curr.get("uv", "N/A")

                forecast_summary = ""
                if forecast_day:
                    max_c = forecast_day.get("maxtemp_c", "N/A")
                    min_c = forecast_day.get("mintemp_c", "N/A")
                    rain_chance = forecast_day.get("daily_chance_of_rain", "0")
                    forecast_summary = f"\n- Today's Forecast: High {max_c}°C / Low {min_c}°C | Chance of rain: {rain_chance}%"

                return (
                    f"### 🌦️ Current Weather for {full_place}\n"
                    f"- Condition: {condition}\n"
                    f"- Temperature: {temp_c}°C ({temp_f}°F)\n"
                    f"- Feels Like: {feels_c}°C ({feels_f}°F)\n"
                    f"- Humidity: {humidity}%\n"
                    f"- Wind: {wind_kph} km/h {wind_dir}\n"
                    f"- UV Index: {uv}"
                    f"{forecast_summary}"
                )
        except Exception as e:
            # Fall through to OpenWeatherMap
            pass

    # 2. Try OpenWeatherMap (Fallback)
    if owm_key and not owm_key.startswith("your_"):
        try:
            encoded_loc = urllib.parse.quote(cleaned_loc)
            url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_loc}&appid={owm_key}&units=metric"
            req = urllib.request.Request(url, headers={"User-Agent": "RiveraGPT/2.0"})
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                place = data.get("name", cleaned_loc)
                country = data.get("sys", {}).get("country", "")
                full_place = f"{place}, {country}" if country else place

                main = data.get("main", {})
                temp_c = main.get("temp", "N/A")
                temp_f = round(temp_c * 9/5 + 32, 1) if isinstance(temp_c, (int, float)) else "N/A"
                feels_c = main.get("feels_like", "N/A")
                feels_f = round(feels_c * 9/5 + 32, 1) if isinstance(feels_c, (int, float)) else "N/A"
                condition = data.get("weather", [{}])[0].get("description", "Unknown").capitalize()
                humidity = main.get("humidity", "N/A")
                wind_spd = data.get("wind", {}).get("speed", "N/A")

                return (
                    f"### 🌦️ Current Weather for {full_place}\n"
                    f"- Condition: {condition}\n"
                    f"- Temperature: {temp_c}°C ({temp_f}°F)\n"
                    f"- Feels Like: {feels_c}°C ({feels_f}°F)\n"
                    f"- Humidity: {humidity}%\n"
                    f"- Wind Speed: {wind_spd} m/s"
                )
        except Exception as e:
            return f"Could not retrieve weather for '{cleaned_loc}': {str(e)}"

    return f"Weather services are currently unconfigured or unavailable for '{cleaned_loc}'."


# ==============================================================================
# 2. LIVE NEWS TOOL (Powered by GNews API)
# ==============================================================================
@tool
def get_latest_news(query: str = "", category: str = "general") -> str:
    """
    Fetch breaking news and top news headlines via GNews.
    If query is provided, searches news articles matching the topic (e.g. 'artificial intelligence', 'space exploration').
    If query is empty, fetches top headlines in the category (general, world, business, technology, entertainment, sports, science, health).
    """
    gnews_key = os.getenv("GNEWS_API_KEY")
    if not gnews_key or gnews_key.startswith("your_"):
        return "Live news is unavailable because GNEWS_API_KEY is not configured."

    try:
        clean_q = query.strip()
        if clean_q:
            encoded_q = urllib.parse.quote(clean_q)
            url = f"https://gnews.io/api/v4/search?q={encoded_q}&lang=en&max=5&apikey={gnews_key}"
            heading = f"### 📰 Latest News for '{clean_q}'"
        else:
            valid_categories = ["general", "world", "business", "technology", "entertainment", "sports", "science", "health"]
            cat = category.lower().strip() if category.lower().strip() in valid_categories else "general"
            url = f"https://gnews.io/api/v4/top-headlines?category={cat}&lang=en&max=5&apikey={gnews_key}"
            heading = f"### 📰 Top News Headlines ({cat.capitalize()})"

        req = urllib.request.Request(url, headers={"User-Agent": "RiveraGPT/2.0"})
        with urllib.request.urlopen(req, timeout=7.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            articles = data.get("articles", [])
            if not articles:
                return f"{heading}\nNo recent articles found."

            output = [heading]
            for idx, a in enumerate(articles[:5], 1):
                title = a.get("title", "Untitled")
                desc = a.get("description", "")
                source = a.get("source", {}).get("name", "Unknown Source")
                pub_date = (a.get("publishedAt") or "")[:10]
                link = a.get("url", "")
                item = f"{idx}. **{title}**\n   - *Source*: {source} | *Date*: {pub_date}\n   - {desc}\n   - [Read Article]({link})"
                output.append(item)

            return "\n\n".join(output)

    except Exception as e:
        return f"Failed to retrieve news: {str(e)}"


# ==============================================================================
# 3. DIRECTIONS & ROUTING TOOL (Powered by HeiGIT / OpenRouteService)
# ==============================================================================
@tool
def get_directions_and_route(start_location: str, end_location: str, mode: str = "driving-car") -> str:
    """
    Calculate driving, cycling, or walking directions, distance, estimated travel time,
    and step-by-step route between two locations using OpenRouteService.
    Modes: 'driving-car' (or 'car'), 'cycling-regular' (or 'bike'), 'foot-walking' (or 'walk').
    Example: get_directions_and_route("Paris, France", "Lyon, France", "driving-car")
    """
    ors_key = (
        os.getenv("OPENROUTESERVICE_API_KEY")
        or os.getenv("ORS_API_KEY")
        or os.getenv("HEIGIT_API_KEY")
    )
    if not ors_key or ors_key.startswith("your_"):
        return "Directions service is unavailable because OPENROUTESERVICE_API_KEY is not configured."

    # Normalize profile mode
    m = mode.lower().strip()
    if "bike" in m or "cycl" in m:
        profile = "cycling-regular"
        mode_label = "🚴 Cycling"
    elif "walk" in m or "foot" in m:
        profile = "foot-walking"
        mode_label = "🚶 Walking"
    else:
        profile = "driving-car"
        mode_label = "🚗 Driving"

    def geocode(loc: str):
        encoded = urllib.parse.quote(loc.strip())
        url = f"https://api.openrouteservice.org/geocode/search?api_key={ors_key}&text={encoded}&size=1"
        req = urllib.request.Request(url, headers={"User-Agent": "RiveraGPT/2.0"})
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            features = data.get("features", [])
            if not features:
                return None, None, None
            coords = features[0]["geometry"]["coordinates"] # [lon, lat]
            name = features[0].get("properties", {}).get("label", loc)
            return coords[0], coords[1], name

    try:
        start_lon, start_lat, start_name = geocode(start_location)
        if start_lon is None:
            return f"Could not find coordinates for origin: '{start_location}'"

        end_lon, end_lat, end_name = geocode(end_location)
        if end_lon is None:
            return f"Could not find coordinates for destination: '{end_location}'"

        # Query Directions API
        dir_url = (
            f"https://api.openrouteservice.org/v2/directions/{profile}"
            f"?api_key={ors_key}&start={start_lon},{start_lat}&end={end_lon},{end_lat}"
        )
        req = urllib.request.Request(dir_url, headers={"User-Agent": "RiveraGPT/2.0"})
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            segment = data["features"][0]["properties"]["segments"][0]

            distance_m = segment.get("distance", 0)
            duration_s = segment.get("duration", 0)

            dist_km = round(distance_m / 1000.0, 1)
            dist_miles = round(distance_m * 0.000621371, 1)

            hours = int(duration_s // 3600)
            mins = int((duration_s % 3600) // 60)
            duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins} mins"

            steps = segment.get("steps", [])
            step_lines = []
            for s in steps[:8]: # Top initial steps
                instruction = s.get("instruction", "")
                step_dist_m = s.get("distance", 0)
                if instruction:
                    step_lines.append(f"- {instruction} ({round(step_dist_m / 1000.0, 2)} km)")

            steps_formatted = "\n".join(step_lines) if step_lines else "Follow main highway route."

            return (
                f"### 🗺️ Route: {start_name} ➔ {end_name}\n"
                f"- Mode: {mode_label}\n"
                f"- Total Distance: {dist_km} km ({dist_miles} miles)\n"
                f"- Estimated Duration: {duration_str}\n\n"
                f"**Key Navigation Steps:**\n{steps_formatted}"
            )

    except Exception as e:
        return f"Failed to compute route from '{start_location}' to '{end_location}': {str(e)}"


# ==============================================================================
# 4. JOB SEARCH TOOL (Powered by Adzuna)
# ==============================================================================
@tool
def search_jobs(query: str, location: str = "", country: str = "us") -> str:
    """
    Search for open job listings, hiring companies, and salary estimates using Adzuna.
    Example: search_jobs("Python Developer", "New York", "us")
    """
    app_key = os.getenv("ADZUNA_APP_KEY") or os.getenv("ADZUNA_API_KEY")
    app_id = os.getenv("ADZUNA_APP_ID")

    if not app_key or app_key.startswith("your_"):
        return "Job search is unavailable because ADZUNA_APP_KEY is not configured."

    if not app_id or app_id.startswith("your_"):
        return (
            f"Adzuna API key is active. To enable live job queries, please add your "
            f"`ADZUNA_APP_ID` from developer.adzuna.com to your `.env` file.\n"
            f"Searching query: '{query}' in '{location or 'all'}'."
        )

    try:
        clean_q = urllib.parse.quote(query.strip())
        country_code = country.strip().lower() or "us"
        url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1?app_id={app_id}&app_key={app_key}&what={clean_q}&results_per_page=5"
        if location.strip():
            url += f"&where={urllib.parse.quote(location.strip())}"

        req = urllib.request.Request(url, headers={"User-Agent": "RiveraGPT/2.0"})
        with urllib.request.urlopen(req, timeout=7.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return f"No current job listings found for '{query}' in '{location or country_code}'."

            output = [f"### 💼 Job Openings for '{query}'"]
            for idx, job in enumerate(results[:5], 1):
                title = job.get("title", "Untitled Position")
                company = job.get("company", {}).get("display_name", "Confidential")
                loc = job.get("location", {}).get("display_name", "Remote / Various")
                salary_min = job.get("salary_min")
                salary_max = job.get("salary_max")
                salary_str = ""
                if salary_min and salary_max:
                    salary_str = f" | Salary: ${salary_min:,.0f} - ${salary_max:,.0f}"
                elif salary_min:
                    salary_str = f" | Salary: from ${salary_min:,.0f}"

                link = job.get("redirect_url", "")
                snippet = job.get("description", "")[:180] + "..."
                output.append(f"{idx}. **{title}**\n   - *Company*: {company} ({loc}){salary_str}\n   - {snippet}\n   - [Apply / View Listing]({link})")

            return "\n\n".join(output)

    except Exception as e:
        return f"Failed to search jobs: {str(e)}"


# ==============================================================================
# 5. WEB SEARCH (Tavily with GNews Fallback)
# ==============================================================================
@tool
def web_search(query: str) -> str:
    """
    Search the web for up-to-date information, current events, or web pages.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key and not api_key.startswith("your_"):
        try:
            from langchain_tavily import TavilySearch
            tavily = TavilySearch(
                max_results=5,
                topic="general",
                search_depth="advanced"
            )
            return str(tavily.invoke({"query": query}))
        except Exception:
            pass

    # Fallback to GNews search if query is news/current info
    news_res = get_latest_news.invoke({"query": query})
    if "Failed" not in news_res and "unavailable" not in news_res:
        return news_res

    return "Web search is currently unavailable. Configure TAVILY_API_KEY or GNEWS_API_KEY."


# ==============================================================================
# 6. MATHEMATICAL CALCULATOR
# ==============================================================================
@tool
def calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions accurately.
    Supports arithmetic, trigonometry, square root, logarithms, power, and rounding.
    Example: 2 + 2, math.sqrt(144), 10 * 5, round(math.pi, 4)
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e
        }
        # Pre-clean expression
        expr = expression.replace("^", "**").strip()
        result = eval(expr, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ==============================================================================
# 7. DOCUMENT RAG & CONVERSATION MEMORY
# ==============================================================================
@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """
    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )


@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """
    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )


@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """
    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )


# All available RiveraGPT assistant tools
tools = [
    get_weather,
    get_latest_news,
    get_directions_and_route,
    search_jobs,
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]

TOOL_MAP = {t.name: t for t in tools}