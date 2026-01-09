import os
import json
import requests

class PersonalityAnalyzer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")

        self.api_key = api_key
        self.model = "llama-3.3-70b-versatile"
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
    def generate_recommendations(
    self,
    review_text: str,
    sentiment: int,
    sentiment_confidence: float,
    personality_scores: dict
):
        sentiment_map = {0: "Négatif", 1: "Neutre", 2: "Positif"}

        prompt = f"""
    You are an HR expert and organizational psychologist.

    Based on:
    1) An employee review
    2) Sentiment analysis result
    3) Big Five personality scores

Generate ACTIONABLE and PRIORITIZED professional recommendations.

Return ONLY valid JSON.
NO markdown.
NO explanations outside JSON.

REVIEW:
\"\"\"{review_text}\"\"\"

SENTIMENT:
- Type: {sentiment_map.get(sentiment, "Neutre")}
- Confidence: {sentiment_confidence:.2f}

BIG FIVE SCORES:
{json.dumps(personality_scores, indent=2)}

FORMAT:
[
  {{
    "icon": "🧠",
    "title": "Short title",
    "action": "Concrete action to take",
    "details": "Why this matters",
    "priority": "URGENT | HAUTE | MOYENNE | BASSE",
    "timeline": "Court terme | Moyen terme | Long terme"
  }}
]
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a senior HR consultant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 500
        }

        try:
            r = requests.post(
                self.url,
                headers=self._headers(),
                json=payload,
                timeout=40
            )

            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            recommendations = json.loads(raw)

            return recommendations

        except Exception as e:
            print("❌ Recommendation error:", e)
            return [
                {
                    "icon": "⚠️",
                    "title": "Analyse complémentaire recommandée",
                    "action": "Discuter avec un manager ou RH pour clarifier la situation",
                    "details": "Les signaux détectés nécessitent une analyse humaine",
                    "priority": "MOYENNE",
                    "timeline": "Court terme"
                }
            ]
