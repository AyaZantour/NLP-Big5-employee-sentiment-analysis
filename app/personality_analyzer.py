import json
from groq import Groq

class PersonalityAnalyzer:
    # Test your API key in personality_analyzer.py __init__
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")
    
        # Test the API key by making a small request
        self.client = Groq(api_key=api_key)
        self.model = "llama3-8b-8192"
    
        # Quick test
        try:
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'TEST OK'"}],
                max_tokens=10
            )
            print(f"✅ Groq API test passed: {test_response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ Groq API test failed: {e}")
    def analyze(self, text: str):
        prompt = f"""
You MUST return ONLY valid JSON.
NO text. NO markdown. NO explanation.

Analyze the following workplace review using the Big Five personality model.
Scores must be REAL numbers between 0.0 and 1.0.

TEXT:
\"\"\"{text}\"\"\"

JSON FORMAT (EXACT):
{{
  "Extraversion": 0.0,
  "Neuroticism": 0.0,
  "Agreeableness": 0.0,
  "Conscientiousness": 0.0,
  "Openness": 0.0
}}
"""

        try:
            # Debug: Print what we're sending
            print(f"🔍 Sending request to Groq...")
            print(f"Model: {self.model}")
            print(f"Text length: {len(text)} chars")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a psychologist. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            
            raw = response.choices[0].message.content.strip()
            print(f"✅ Raw response received: {raw[:100]}...")
            
        except Exception as e:
            print(f"❌ Groq API call failed: {type(e).__name__}: {e}")
            # Return fallback so app continues
            return {
                "scores": {
                    "Extraversion": 0.5,
                    "Névrosisme": 0.5,
                    "Agréabilité": 0.5,
                    "Conscience": 0.5,
                    "Ouverture": 0.5
                },
                "model_used": self.model,
                "analysis": f"API Error: {str(e)[:100]}"
            }