import requests

def get_weather(city: str) -> str:
    api_key = "YOUR_API_KEY"  # get from openweathermap.org
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=e71b5f18557da86f35fdf0cf5b01525c&units=metric"

    response = requests.get(url)
    data = response.json()

    if data.get("main"):
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The weather in {city} is {temp}°C with {desc}."
    else:
        return "Weather data not found."
