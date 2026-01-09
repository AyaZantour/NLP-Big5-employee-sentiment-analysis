# # app/personality_analyzer.py - CLEAN VERSION
# import os
# from groq import Groq
# import json
# import re
# import streamlit as st



# class PersonalityAnalyzer:
#     """Big Five Personality Analyzer using Groq API"""
    
#     def __init__(self, api_key=None):
#         # Get API key from environment or parameter
#         # self.api_key = api_key or os.getenv('GROQ_API_KEY')
#         self.api_key = api_key or st.secrets["GROQ_API_KEY"]

        
#         if not self.api_key:
#             raise ValueError("GROQ_API_KEY not found. Please set it in .env file")
        
#         # Initialize Groq client
#         self.client = Groq(api_key=self.api_key)
        
#         self.model = "llama3-8b-8192"  # ← This model works better with JSON



#     def analyze(self, text):
#         prompt = f"""
#     Return ONLY valid JSON.
#     No explanation. No markdown. No text.

#     Analyze the following work review and estimate Big Five personality traits.
#     Scores must be between 0 and 1.

#     TEXT:
#     {text}

#     JSON FORMAT:
#     {{
#     "Extraversion": 0.0,
#     "Neuroticism": 0.0,
#     "Agreeableness": 0.0,
#     "Conscientiousness": 0.0,
#     "Openness": 0.0
#     }}
#     """

#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a psychologist. You must output valid JSON only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.0,
#                 max_tokens=150
#             )

#             content = response.choices[0].message.content.strip()

#             # 🔒 STRICT JSON PARSE
#             scores_raw = json.loads(content)

#             # 🔧 Clamp values to [0,1]
#             def clamp(x): 
#                 return max(0.0, min(1.0, float(x)))

#             scores_fr = {
#                 "Extraversion": clamp(scores_raw.get("Extraversion", 0.5)),
#                 "Névrosisme": clamp(scores_raw.get("Neuroticism", 0.5)),
#                 "Agréabilité": clamp(scores_raw.get("Agreeableness", 0.5)),
#                 "Conscience": clamp(scores_raw.get("Conscientiousness", 0.5)),
#                 "Ouverture": clamp(scores_raw.get("Openness", 0.5))
#             }

#             return {
#                 "scores": scores_fr,
#                 "model_used": self.model,
#                 "analysis": None
#             }

#         except Exception as e:
#             print("Groq personality error:", e)

#             # Safe fallback
#             return {
#                 "scores": {
#                     "Extraversion": 0.5,
#                     "Névrosisme": 0.5,
#                     "Agréabilité": 0.5,
#                     "Conscience": 0.5,
#                     "Ouverture": 0.5
#                 },
#                 "model_used": self.model,
#                 "analysis": None
#             }
  





# app/personality_analyzer.py - FIXED VERSION
import os
# Disable proxy variables injected by Streamlit Cloud
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(var, None)

from groq import Groq
import json
# import streamlit as st

# class PersonalityAnalyzer:
#     """Big Five Personality Analyzer using Groq API"""
    
#     def __init__(self, api_key=None):
#         # Get API key from Streamlit secrets or environment
#         self.api_key = api_key or st.secrets.get("GROQ_API_KEY") or os.getenv('GROQ_API_KEY')
        
#         if not self.api_key:
#             raise ValueError("GROQ_API_KEY not found. Please set it in .streamlit/secrets.toml or .env")
        
#         # Initialize Groq client
#         self.client = Groq(api_key=self.api_key)
        
#         # Use a more capable model
#         self.model = "llama-3.3-70b-versatile"  # Better than 8b for analysis

class PersonalityAnalyzer:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        
        # IMPORTANT: Clean environment before creating Groq client
        import os
        
        # Remove all proxy environment variables
        proxy_vars = [
            'HTTP_PROXY', 'HTTPS_PROXY', 
            'http_proxy', 'https_proxy',
            'ALL_PROXY', 'all_proxy',
            'STREAMLIT_PROXY', 'STREAMLIT_SERVER_PROXY'
        ]
        
        saved_proxies = {}
        for var in proxy_vars:
            if var in os.environ:
                saved_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Import Groq here, after environment cleanup
            from groq import Groq
            
            # Create client with minimal parameters
            try:
                # Try with just api_key
                self.client = Groq(api_key=api_key)
            except TypeError:
                # If that fails, inspect what parameters are accepted
                import inspect
                sig = inspect.signature(Groq.__init__)
                params = list(sig.parameters.keys())
                
                # Remove 'self' and 'api_key'
                params = [p for p in params if p not in ['self', 'api_key']]
                
                # Create kwargs without 'proxies'
                kwargs = {}
                for param in params:
                    if param != 'proxies':
                        kwargs[param] = None
                
                self.client = Groq(api_key=api_key, **kwargs)
        
        finally:
            # Restore environment variables
            for var, value in saved_proxies.items():
                os.environ[var] = value
        
        self.model = "llama-3.3-70b-versatile"    
    
    def analyze(self, text):
        """Analyze text and return Big Five personality scores"""
        
        # 🔥 MUCH BETTER PROMPT - Gives context and examples
        prompt = f"""You are an expert organizational psychologist specializing in the Big Five personality model.

Analyze this workplace review and estimate the author's Big Five personality traits based on their language, tone, and content.

REVIEW TEXT:
\"\"\"{text}\"\"\"

SCORING GUIDE (0.0 to 1.0):

**Extraversion** (0 = very introverted, 1 = very extroverted)
- HIGH: Mentions team activities, social events, collaboration, networking
- LOW: Prefers working alone, mentions quiet workspace, independent work

**Neuroticism** (0 = very stable, 1 = very neurotic/anxious)
- HIGH: Complains about stress, anxiety, work-life balance issues, burnout
- LOW: Mentions stability, calmness, handles pressure well

**Agreeableness** (0 = competitive/critical, 1 = cooperative/trusting)
- HIGH: Praises teamwork, supportive culture, friendly colleagues
- LOW: Critical tone, mentions conflicts, competitive environment

**Conscientiousness** (0 = spontaneous/careless, 1 = organized/disciplined)
- HIGH: Values structure, processes, deadlines, organization, planning
- LOW: Mentions flexibility, spontaneity, dislikes rigid rules

**Openness** (0 = traditional/practical, 1 = creative/curious)
- HIGH: Mentions innovation, learning, new technologies, creative projects
- LOW: Values routine, stability, proven methods, tradition

Return ONLY valid JSON with scores between 0.0 and 1.0:
{{"Extraversion": 0.0, "Neuroticism": 0.0, "Agreeableness": 0.0, "Conscientiousness": 0.0, "Openness": 0.0}}

Be confident in your estimates - avoid defaulting to 0.5 unless truly neutral."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert psychologist. Analyze personality traits from text. Return ONLY valid JSON with scores 0.0-1.0. Be confident - avoid middle values unless necessary."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Slightly more creative than 0.0
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            scores_raw = json.loads(content)
            
            # Clamp values to [0,1] and convert to French
            def clamp(x):
                return max(0.0, min(1.0, float(x)))
            
            scores_fr = {
                "Extraversion": clamp(scores_raw.get("Extraversion", 0.5)),
                "Névrosisme": clamp(scores_raw.get("Neuroticism", 0.5)),
                "Agréabilité": clamp(scores_raw.get("Agreeableness", 0.5)),
                "Conscience": clamp(scores_raw.get("Conscientiousness", 0.5)),
                "Ouverture": clamp(scores_raw.get("Openness", 0.5))
            }
            
            # Generate a brief analysis
            analysis = self._generate_analysis(scores_fr)
            
            return {
                "scores": scores_fr,
                "model_used": self.model,
                "analysis": analysis
            }
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {content}")
            # Fallback to neutral scores
            return self._fallback_response()
        
        except Exception as e:
            print(f"Groq API error: {e}")
            return self._fallback_response()
    
    def _generate_analysis(self, scores):
        """Generate a brief personality analysis"""
        
        # Find highest and lowest traits
        sorted_traits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        highest = sorted_traits[0]
        lowest = sorted_traits[-1]
        
        analysis = f"""**Profil dominant:** {highest[0]} élevé ({highest[1]:.0%})

**Traits principaux:**
"""
        
        # Add interpretations for high scores
        for trait, score in sorted_traits[:3]:
            if score > 0.6:
                if trait == "Extraversion":
                    analysis += "- 🗣️ Personne sociable et énergique\n"
                elif trait == "Névrosisme":
                    analysis += "- 😰 Sensibilité émotionnelle élevée\n"
                elif trait == "Agréabilité":
                    analysis += "- 🤝 Coopératif et empathique\n"
                elif trait == "Conscience":
                    analysis += "- 📋 Organisé et discipliné\n"
                elif trait == "Ouverture":
                    analysis += "- 💡 Créatif et curieux\n"
        
        return analysis
    
    # def _fallback_response(self):
    #     """Return neutral scores as fallback"""
    #     return {
    #         "scores": {
    #             "Extraversion": 0.5,
    #             "Névrosisme": 0.5,
    #             "Agréabilité": 0.5,
    #             "Conscience": 0.5,
    #             "Ouverture": 0.5
    #         },
    #         "model_used": self.model,
    #         "analysis": "⚠️ Analyse par défaut (erreur API)"
    #     }
    
    def generate_recommendations(self, review_text, sentiment, sentiment_confidence, personality_scores):
        """
        Generate personalized recommendations using AI based on:
        - Review content
        - Sentiment analysis
        - Personality traits
        """
        
        # Build personality summary
        traits_summary = "\n".join([
            f"- {trait}: {score:.0%}" 
            for trait, score in personality_scores.items()
        ])
        
        # Determine sentiment label
        if sentiment == 0:
            sentiment_label = "Négatif"
        elif sentiment == 1:
            sentiment_label = "Neutre"
        else:
            sentiment_label = "Positif"
        
        prompt = f"""Vous êtes un coach professionnel expert en développement de carrière et psychologie organisationnelle.

CONTEXTE DE L'EMPLOYÉ:

Review complète:
\"\"\"{review_text}\"\"\"

Analyse de sentiment:
- Sentiment dominant: {sentiment_label}
- Niveau de confiance: {sentiment_confidence:.1%}

Profil de personnalité Big Five:
{traits_summary}

TÂCHE:
Générez exactement 4 recommandations ACTIONABLES et PERSONNALISÉES pour cet employé.

RÈGLES STRICTES:
1. Chaque recommandation doit être spécifique au contexte de cette personne
2. Priorisez les actions qui auront le plus d'impact immédiat
3. Soyez concret: donnez des actions précises, pas des généralités
4. Adaptez le ton selon le sentiment (urgent si négatif, encourageant si positif)
5. Tenez compte des traits de personnalité dominants

STRUCTURE JSON REQUISE:
{{
  "recommendations": [
    {{
      "priority": "URGENT" ou "HAUTE" ou "MOYENNE",
      "title": "Titre court (5-7 mots max)",
      "action": "Action spécifique à réaliser (1 phrase)",
      "details": "Explication contextuelle (2-3 phrases)",
      "timeline": "Timeframe précis (ex: Cette semaine, Sous 3 jours)"
    }},
    ... (4 recommandations au total)
  ]
}}

EXEMPLES DE BONNES RECOMMANDATIONS:
- "Documentez 3 incidents précis avec dates avant votre entretien RH vendredi"
- "Proposez une réunion café avec votre manager pour discuter de 2 projets qui vous intéressent"
- "Inscrivez-vous à la formation 'Communication assertive' disponible sur votre LMS interne"

Retournez UNIQUEMENT le JSON, sans texte additionnel."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un coach professionnel expert. Tu génères des recommandations personnalisées actionnables. Réponds UNIQUEMENT en JSON valide."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Slight creativity for varied recommendations
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean markdown
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            result = json.loads(content)
            
            # Validate structure
            if "recommendations" not in result:
                raise ValueError("Missing 'recommendations' key")
            
            recommendations = result["recommendations"]
            
            # Add icons based on priority
            icon_map = {
                "URGENT": "🚨",
                "HAUTE": "⚡",
                "MOYENNE": "🎯",
                "BASSE": "💡"
            }
            
            for rec in recommendations:
                priority = rec.get("priority", "MOYENNE")
                rec["icon"] = icon_map.get(priority, "💡")
            
            return recommendations
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {content}")
            return self._fallback_recommendations(sentiment_label)
        
        except Exception as e:
            print(f"Groq recommendation error: {e}")
            return self._fallback_recommendations(sentiment_label)
    
    def _fallback_recommendations(self, sentiment_label):
        """Fallback recommendations if AI fails"""
        
        if sentiment_label == "Négatif":
            return [
                {
                    "icon": "🚨",
                    "priority": "URGENT",
                    "title": "Entretien avec votre manager",
                    "action": "Planifiez une discussion cette semaine",
                    "details": "Préparez 3 problèmes concrets avec des exemples précis.",
                    "timeline": "Cette semaine"
                },
                {
                    "icon": "📋",
                    "priority": "HAUTE",
                    "title": "Documentation des incidents",
                    "action": "Listez les problèmes avec dates et contexte",
                    "details": "Gardez une trace objective pour les discussions futures.",
                    "timeline": "Sous 3 jours"
                }
            ]
        
        elif sentiment_label == "Neutre":
            return [
                {
                    "icon": "🎯",
                    "priority": "HAUTE",
                    "title": "Objectifs de développement",
                    "action": "Définissez 2-3 objectifs SMART pour ce trimestre",
                    "details": "Discutez-les avec votre manager lors de votre prochaine 1-on-1.",
                    "timeline": "Cette semaine"
                }
            ]
        
        else:  # Positif
            return [
                {
                    "icon": "🚀",
                    "priority": "MOYENNE",
                    "title": "Leadership et mentorat",
                    "action": "Proposez de mentorer un collègue junior",
                    "details": "Partagez votre expérience positive et contribuez à la culture.",
                    "timeline": "Ce mois-ci"
                }
            ]
        
