import os
import json
import requests

class PersonalityAnalyzer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")

        self.api_key = api_key
        self.model = "llama3-8b-8192"
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        # 🔍 quick test
        self._test_api()

    def _test_api(self):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5
        }

        r = requests.post(
            self.url,
            headers=self._headers(),
            json=payload,
            timeout=20
        )

        if r.status_code != 200:
            raise RuntimeError(f"Groq API test failed: {r.text}")

        print("✅ Groq REST API OK")

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def analyze(self, text: str):
        prompt = f"""
Return ONLY valid JSON.
Numbers between 0.0 and 1.0.

TEXT:
\"\"\"{text}\"\"\"

FORMAT:
{{
  "Extraversion": 0.0,
  "Neuroticism": 0.0,
  "Agreeableness": 0.0,
  "Conscientiousness": 0.0,
  "Openness": 0.0
}}
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a psychologist."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 150
        }

        try:
            r = requests.post(
                self.url,
                headers=self._headers(),
                json=payload,
                timeout=30
            )

            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            scores = json.loads(raw)

            return {
                "scores": scores,
                "model_used": self.model
            }

        except Exception as e:
            print("❌ Groq error:", e)
            return {
                "scores": {
                    "Extraversion": 0.5,
                    "Neuroticism": 0.5,
                    "Agreeableness": 0.5,
                    "Conscientiousness": 0.5,
                    "Openness": 0.5
                },
                "model_used": self.model,
                "analysis": "Fallback due to Groq error"
            }
