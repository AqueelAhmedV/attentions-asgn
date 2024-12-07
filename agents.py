from typing import Dict, Any, Optional
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from memory import MemoryAgent
import aiohttp
import json
from ollama_utils import OllamaLLM

class WeatherAgent:
    def __init__(self, memory_agent: MemoryAgent, weather_api_key: str):
        """Initialize Weather Agent with Memory integration"""
        self.memory_agent = memory_agent
        self.weather_api_key = weather_api_key
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM("mistral")
        
        # Set the LLM as default
        Settings.llm = self.llm
        
        # Initialize agent
        self.agent = ReActAgent.from_tools(
            tools=[
                self.get_weather,
                self.get_user_location_preferences
            ],
            llm=self.llm,
            verbose=True
        )

    async def get_weather(self, city: str) -> Dict[str, Any]:
        """Get weather information for a specific city"""
        async with aiohttp.ClientSession() as session:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "description": data["weather"][0]["description"],
                        "wind_speed": data["wind"]["speed"]
                    }
                else:
                    return {"error": f"Could not fetch weather for {city}"}

    async def get_user_location_preferences(self, username: str) -> Dict[str, Any]:
        """Get user's location preferences from memory"""
        preferences = await self.memory_agent.get_user_preferences(username)
        return {
            "visited_cities": preferences.get("visited_cities", []),
            "interests": [i for i in preferences.get("interests", []) 
                         if "city" in i.get("type", "").lower()]
        }

    async def process_weather_query(self, username: str, query: str) -> str:
        """Process a weather-related query using both weather data and user memory"""
        response = await self.agent.aquery(
            f"""
            Context: User {username} is asking about weather.
            Their Query: {query}
            
            Use the available tools to:
            1. Check their location preferences
            2. Get relevant weather data
            
            Provide a helpful response considering both their preferences and the weather data.
            """
        )
        return str(response)

    async def suggest_weather_based_activities(self, username: str, city: str) -> str:
        """Suggest activities based on weather and user preferences"""
        weather_data = await self.get_weather(city)
        user_prefs = await self.get_user_location_preferences(username)
        
        response = await self.agent.aquery(
            f"""
            Weather in {city}:
            {json.dumps(weather_data, indent=2)}
            
            User Preferences:
            {json.dumps(user_prefs, indent=2)}
            
            Suggest activities considering:
            1. The current weather conditions
            2. The user's previous travel history
            3. Their stored interests
            """
        )
        return str(response) 