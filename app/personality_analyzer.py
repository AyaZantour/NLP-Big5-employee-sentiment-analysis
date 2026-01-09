import os
import json
from groq import Groq

class PersonalityAnalyzer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")

        # 🚨 FIX STREAMLIT CLOUD PROXY ISSUE
        for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            os.environ.pop(var, None)

        self.client = Groq(api_key=api_key)
        self.model = "llama3-8b-8192"

        # 🔍 Test API key
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5
            )
            print("✅ Groq API OK")
        except Exception as e:
            raise RuntimeError(f"Groq API test failed: {e}")

    def analyze(self, text: str):
        prompt = f"""
Return ONLY valid JSON.
Numbers between 0.0 and 1.0.
NO explanation.

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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a psychologist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=120
            )

            raw = response.choices[0].message.content.strip()
            scores = json.loads(raw)

            return {
                "scores": scores,
                "model_used": self.model
            }

        except Exception as e:
            print(f"❌ Groq error: {e}")
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
