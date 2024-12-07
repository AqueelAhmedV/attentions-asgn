import requests
import json
import config

class LLMClient:
    def __init__(self, base_url=config.OLLAMA_BASE_URL):
        self.base_url = base_url
        
    async def get_completion(self, prompt, system_prompt=None):
        try:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": "mistral",
                    "messages": messages,
                    "stream": False
                }
            )
            return response.json()['message']['content']
        except Exception as e:
            print(f"Error in LLM completion: {e}")
            return "I'm sorry, I encountered an error processing your request." 