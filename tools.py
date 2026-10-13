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
                    f"{forecast_summary}\n\n*Provider: WeatherAPI*"
                )
        except Exception:
            # Fall through to OpenWeatherMap or Open-Meteo
            pass

    # 2. Try OpenWeatherMap (Secondary authenticated fallback)
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
                    f"- Wind Speed: {wind_spd} m/s\n\n*Provider: OpenWeatherMap*"
                )
        except Exception:
            pass

    # 3. Open-Meteo (Tertiary Zero-Key Fallback — Free, Global, No Credentials Required)
    try:
        encoded_loc = urllib.parse.quote(cleaned_loc)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_loc}&count=1&language=en"
        geo_req = urllib.request.Request(geo_url, headers={"User-Agent": "RiveraGPT/2.0"})
        with urllib.request.urlopen(geo_req, timeout=5.0) as resp:
            geo_data = json.loads(resp.read().decode("utf-8"))
            results = geo_data.get("results", [])
            if results:
                target = results[0]
                lat = target.get("latitude")
                lon = target.get("longitude")
                name = target.get("name", cleaned_loc)
                admin1 = target.get("admin1", "")
                country = target.get("country", "")
                full_place = ", ".join([p for p in [name, admin1, country] if p])

                forecast_url = (
                    f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m,weather_code"
                    f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto"
                )
                fc_req = urllib.request.Request(forecast_url, headers={"User-Agent": "RiveraGPT/2.0"})
                with urllib.request.urlopen(fc_req, timeout=6.0) as fc_resp:
                    fc_data = json.loads(fc_resp.read().decode("utf-8"))
                    curr = fc_data.get("current", {})
                    daily = fc_data.get("daily", {})

                    wmo_codes = {
                        0: "Clear sky ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
                        45: "Foggy 🌫️", 48: "Depositing rime fog 🌫️",
                        51: "Light drizzle 🌦️", 53: "Moderate drizzle 🌦️", 55: "Dense drizzle 🌧️",
                        61: "Slight rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
                        71: "Slight snow 🌨️", 73: "Moderate snow 🌨️", 75: "Heavy snow ❄️",
                        80: "Rain showers 🌦️", 81: "Moderate rain showers 🌧️", 82: "Violent rain showers ⛈️",
                        95: "Thunderstorm ⛈️", 96: "Thunderstorm with slight hail ⛈️", 99: "Thunderstorm with heavy hail ⛈️"
                    }

                    w_code = curr.get("weather_code", 0)
                    condition = wmo_codes.get(w_code, "Fair 🌤️")
                    temp_c = curr.get("temperature_2m", "N/A")
                    temp_f = round(temp_c * 9/5 + 32, 1) if isinstance(temp_c, (int, float)) else "N/A"
                    feels_c = curr.get("apparent_temperature", "N/A")
                    feels_f = round(feels_c * 9/5 + 32, 1) if isinstance(feels_c, (int, float)) else "N/A"
                    humidity = curr.get("relative_humidity_2m", "N/A")
                    wind_kph = curr.get("wind_speed_10m", "N/A")

                    forecast_summary = ""
                    if daily.get("temperature_2m_max"):
                        max_c = daily["temperature_2m_max"][0]
                        min_c = daily["temperature_2m_min"][0]
                        rain_chance = daily.get("precipitation_probability_max", [0])[0]
                        forecast_summary = f"\n- Today's Forecast: High {max_c}°C / Low {min_c}°C | Rain probability: {rain_chance}%"

                    return (
                        f"### 🌦️ Current Weather for {full_place}\n"
                        f"- Condition: {condition}\n"
                        f"- Temperature: {temp_c}°C ({temp_f}°F)\n"
                        f"- Feels Like: {feels_c}°C ({feels_f}°F)\n"
                        f"- Humidity: {humidity}%\n"
                        f"- Wind Speed: {wind_kph} km/h"
                        f"{forecast_summary}\n\n*Provider: Open-Meteo (Zero-Key)*"
                    )
    except Exception as e:
        return f"Could not retrieve weather for '{cleaned_loc}': {str(e)}"

    return f"Weather services are currently unavailable for '{cleaned_loc}'. Please verify the spelling or try another city."


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
def search_jobs(query: str, location: str = "", country: str = "us", experience_level: str = "") -> str:
    """
    Search for verified open job listings, hiring companies, and salary estimates using Adzuna.
    Example: search_jobs("Python Developer", "New York", "us", "entry-level")
    """
    clean_q = (query or "").strip()
    clean_loc = (location or "").strip()

    # Slot-filling guidance: check if role is generic or missing
    generic_terms = ["job", "jobs", "internship", "internships", "hiring", "work", "career", "openings"]
    if not clean_q or clean_q.lower() in generic_terms:
        return (
            "To find the best job listings for you, please let me know:\n"
            "1. **Job Role** (e.g. *AI Engineer*, *Frontend Developer*, *Data Scientist*)\n"
            "2. **Location** (e.g. *San Francisco*, *London*, *Hyderabad*, or *Remote*)\n"
            "3. **Experience Level** (e.g. *Internship*, *Entry-Level*, *Mid-Level*, *Senior*)"
        )

    app_key = os.getenv("ADZUNA_APP_KEY") or os.getenv("ADZUNA_API_KEY")
    app_id = os.getenv("ADZUNA_APP_ID")

    if not app_key or app_key.startswith("your_"):
        return "Job search is unavailable because ADZUNA_APP_KEY is not configured in your settings."

    if not app_id or app_id.startswith("your_"):
        loc_str = clean_loc if clean_loc else "Remote / Any"
        exp_str = f" ({experience_level})" if experience_level else ""
        return (
            f"### 💼 Job Search Setup Required\n"
            f"Your Adzuna API Key is active. To enable live job queries, please add your **`ADZUNA_APP_ID`** from [developer.adzuna.com](https://developer.adzuna.com) in your Settings ⚙️ menu or `.env` file.\n\n"
            f"- **Target Role**: {clean_q}{exp_str}\n"
            f"- **Location**: {loc_str}\n\n"
            f"*RiveraGPT adheres to strict factual verification standards and never fabricates simulated or fake job postings.*"
        )

    try:
        encoded_q = urllib.parse.quote(clean_q)
        country_code = country.strip().lower() or "us"
        url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1?app_id={app_id}&app_key={app_key}&what={encoded_q}&results_per_page=10"
        if clean_loc:
            url += f"&where={urllib.parse.quote(clean_loc)}"

        req = urllib.request.Request(url, headers={"User-Agent": "RiveraGPT/2.0"})
        with urllib.request.urlopen(req, timeout=7.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return f"No verified job listings currently found for '{clean_q}' in '{clean_loc or country_code}'."

            # Deduplicate by job URL or title+company
            seen_keys = set()
            unique_jobs = []
            for job in results:
                url_key = job.get("redirect_url")
                title_key = (job.get("title", "").strip().lower(), job.get("company", {}).get("display_name", "").strip().lower())
                if url_key and url_key in seen_keys:
                    continue
                if title_key in seen_keys:
                    continue
                if url_key:
                    seen_keys.add(url_key)
                seen_keys.add(title_key)
                unique_jobs.append(job)
                if len(unique_jobs) >= 5:
                    break

            loc_header = f" in {clean_loc}" if clean_loc else ""
            output = [f"### 💼 Verified Job Openings for '{clean_q}'{loc_header}"]
            for idx, job in enumerate(unique_jobs, 1):
                title = job.get("title", "Untitled Position")
                company = job.get("company", {}).get("display_name", "Confidential")
                loc = job.get("location", {}).get("display_name", clean_loc or "Remote / Various")
                created = (job.get("created") or "")[:10]
                salary_min = job.get("salary_min")
                salary_max = job.get("salary_max")
                salary_str = ""
                if salary_min and salary_max:
                    salary_str = f" | 💵 ${salary_min:,.0f} - ${salary_max:,.0f}/yr"
                elif salary_min:
                    salary_str = f" | 💵 From ${salary_min:,.0f}/yr"

                date_str = f" | 📅 Posted {created}" if created else ""
                link = job.get("redirect_url", "")
                snippet = job.get("description", "")[:180].replace("\n", " ").strip() + "..."
                output.append(
                    f"{idx}. **{title}**\n"
                    f"   - **Company**: {company} ({loc}){salary_str}{date_str}\n"
                    f"   - {snippet}\n"
                    f"   - [🔗 Apply / View Official Posting]({link})"
                )

            return "\n\n".join(output)

    except Exception as e:
        return f"Failed to search jobs via Adzuna: {str(e)}"


# ==============================================================================
# 5. IMAGE GENERATION TOOL (FLUX.1 Photorealistic AI)
# ==============================================================================
@tool
def generate_ai_image(prompt: str) -> str:
    """
    Generate high-resolution photorealistic AI images, illustrations, logos, or visual concepts from a prompt.
    Powered by FLUX.1 state-of-the-art diffusion engine.
    Example: generate_ai_image("a futuristic agricultural water management system logo, minimalist vector")
    """
    try:
        from image_service import generate_image
        res = generate_image(prompt)
        if res.get("success"):
            return res.get("markdown", f"![{prompt}]({res.get('url')})")
        return f"Image generation failed: {res.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Image generation error: {str(e)}"


# ==============================================================================
# 6. WEB SEARCH (Tavily with GNews Fallback)
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
# 7. MATHEMATICAL CALCULATOR
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
# 8. DOCUMENT RAG & CONVERSATION MEMORY
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
    generate_ai_image,
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]

TOOL_MAP = {t.name: t for t in tools}