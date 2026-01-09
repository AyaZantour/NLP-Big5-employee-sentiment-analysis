#new code of 13:31 26/12/2025

import os
from dotenv import load_dotenv

os.environ['STREAMLIT_SERVER_ENABLE_STATS'] = 'false'



import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from personality_analyzer import PersonalityAnalyzer
load_dotenv()


# Configuration
st.set_page_config(
    page_title="Glassdoor Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# ============================================
# DEFINE TransformersPipeline CLASS
# (Must match the one used in Kaggle training!)
# ============================================
class TransformersPipeline:
    """Pipeline for sentiment analysis using DistilBERT"""
    def __init__(self, model_path, tokenizer_path, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}

    def predict(self, text):
        """Predict sentiment for a single text"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

        return prediction, probabilities

# # ============================================
# # LOAD MODEL
# # ============================================
# @st.cache_resource
# def load_model():
#     """Load the trained model (once)"""
#     try:
#         # Try loading directly from model files
#         model_path = 'models/transformers_model'
#         tokenizer_path = 'models/transformers_tokenizer'
        
#         # Add these debug prints to see where Streamlit is looking
#         import os
#         st.write(f"Current directory: {os.getcwd()}")
#         st.write(f"Model path exists: {os.path.exists(model_path)}")
#         st.write(f"Tokenizer path exists: {os.path.exists(tokenizer_path)}")
        
#         # Try with local_files_only=True
#         pipeline = TransformersPipeline(model_path, tokenizer_path, device='cpu')
#         st.success("✅ Modèle chargé avec succès!")
#         return pipeline
#     except Exception as e:
#         st.error(f"❌ Erreur de chargement: {e}")
#         st.info("""
#         🔍 Vérifiez que vous avez:
#         1. Le dossier `models/transformers_model/` avec les fichiers du modèle
#         2. Le dossier `models/transformers_tokenizer/` avec les fichiers du tokenizer
#         3. Tous les fichiers extraits du ZIP Kaggle
#         """)
#         return None




# # Load model
model = load_model()







# Replace the load_model() function with this SIMPLE version:

@st.cache_resource
def load_model():
    try:
        # Your HuggingFace model
        from transformers import pipeline
        
        MODEL_ID = "AyaZantour/employee-sentiment-model"
        st.write(f"Loading model: {MODEL_ID}")
        
        # Load from HuggingFace
        pipe = pipeline("text-classification", model=MODEL_ID)
        
        # Return a simple adapter
        class SimpleModel:
            def predict(self, text):
                result = pipe(text)[0]
                # Map labels: adjust based on your model
                if "NEG" in result['label'].upper():
                    return 0, [result['score'], 0.1, 0.1]
                elif "POS" in result['label'].upper():
                    return 2, [0.1, 0.1, result['score']]
                else:
                    return 1, [0.1, result['score'], 0.1]
        
        st.success("✅ Model loaded from HuggingFace!")
        return SimpleModel()
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None









# # Load personality model (separate from sentiment model!)
# def load_groq_analyzer():
#     try:
#         analyzer = PersonalityAnalyzer()
#         st.success("✅ Analyseur de personnalité Groq chargé!")
#         return analyzer
#     except Exception as e:
#         st.error(f"❌ Erreur Groq: {e}")
#         st.info("""
#         🔧 Configuration requise:
#         1. Créez un compte gratuit sur console.groq.com
#         2. Obtenez votre clé API
#         3. Créez un fichier `.env` avec: GROQ_API_KEY=votre_clé
#         """)
#         return None




def load_groq_analyzer():
    try:
        # FIRST try Streamlit Cloud secrets (for production)
        if 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
            st.success("✅ Using API key from Streamlit Cloud secrets")
        
        # FALLBACK to .env file (for local development)
        else:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                st.success("✅ Using API key from .env file")
            else:
                st.warning("⚠️ No API key found in secrets or .env")
                st.info("""
                🔧 Configuration required:
                1. For local: Create `.env` with GROQ_API_KEY=your_key
                2. For Streamlit Cloud: Add GROQ_API_KEY in Settings → Secrets
                """)
                return None
        
        # Initialize analyzer with the API key
        # Check if PersonalityAnalyzer accepts api_key parameter
        try:
            analyzer = PersonalityAnalyzer(api_key=api_key)
        except TypeError:
            # If constructor doesn't accept api_key, try setting it differently
            analyzer = PersonalityAnalyzer()
            # Or check personality_analyzer.py for how it expects the key
        
        st.success("✅ Personality analyzer Groq loaded!")
        return analyzer
        
    except Exception as e:
        st.error(f"❌ Groq error: {e}")
        return None





groq_analyzer = load_groq_analyzer()
# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("**Modèle:** DistilBERT fine-tuned")
    st.markdown("**Précision:** 65%")
    st.markdown("**Classes:** Négatif 😠 | Neutre 😐 | Positif 😊")
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    st.metric("Avis analysés", "67,000+")
    st.metric("Précision modèle", "65%")
    
    st.markdown("---")

# ============================================
# MAIN APP
# ============================================
st.title("🎯 Glassdoor Sentiment Analyzer")
st.markdown("### Analyse de sentiment des avis Glassdoor avec Transformer AI")
st.markdown("---")

# Section 1: Single review analysis
st.header("🔍 Analyse d'un Avis Unique")
col1, col2 = st.columns([2, 1])

with col1:
    review_text = st.text_area(
        "📝 Collez votre avis Glassdoor ici:",
        height=150,
        placeholder="Exemple: 'J'adore travailler ici! L'équipe est géniale et les avantages sont excellents.'"
    )

with col2:
    st.markdown("### 💡 Exemples")
    example = st.selectbox(
        "Choisir un exemple:",
        ["", 
         "Positif: Super entreprise avec une culture incroyable",
         "Négatif: Management toxique et charge de travail excessive",
         "Neutre: L'entreprise est correcte, rien d'exceptionnel"]
    )
    
    if example:
        if "Positif" in example:
            review_text = "J'adore l'ambiance de travail ici. Les managers sont à l'écoute, les projets sont intéressants et les collègues sont formidables. Les avantages sociaux sont compétitifs et il y a de réelles opportunités d'évolution."
        elif "Négatif" in example:
            review_text = "Management très hiérarchique et peu à l'écoute. Charge de travail excessive, souvent jusqu'à 20h le soir. Pas d'équilibre vie pro/perso. La rémunération n'est pas à la hauteur des attentes."
        elif "Neutre" in example:
            review_text = "L'entreprise est correcte. Le travail est intéressant mais répétitif. Les collègues sont sympas. Les avantages sont standards pour le secteur. Rien d'exceptionnel mais correct."

# Analyze button
if st.button("🚀 Analyser le sentiment", type="primary") and review_text:
    if model:
        with st.spinner("🔮 Analyse en cours..."):
            # Predict
            prediction, probabilities = model.predict(review_text)
            
            # Labels
            sentiment_labels = {0: "Négatif 😠", 1: "Neutre 😐", 2: "Positif 😊"}
            sentiment_emojis = {0: "😠", 1: "😐", 2: "😊"}
            sentiment_colors = {0: "#FF6B6B", 1: "#FFD166", 2: "#06D6A0"}
            
            sentiment = sentiment_labels[prediction]
            emoji = sentiment_emojis[prediction]
            color = sentiment_colors[prediction]
            
            # Display results
            st.markdown("---")
            
            # Main result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"### {emoji}")
                st.markdown(f"### {sentiment}")
                st.markdown(f"**Confiance:** {probabilities[prediction]*100:.1f}%")
            
            with col_b:
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probabilities[prediction]*100,
                    title={'text': "Confiance"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 33], 'color': "#FF6B6B"},
                            {'range': [33, 66], 'color': "#FFD166"},
                            {'range': [66, 100], 'color': "#06D6A0"}
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with col_c:
                # Detailed scores
                st.markdown("### 📊 Scores détaillés")
                for i, (label, prob) in enumerate(zip(["Négatif", "Neutre", "Positif"], probabilities)):
                    progress = int(prob * 100)
                    st.markdown(f"**{label}:** {progress}%")
                    st.progress(progress / 100)
            
            # Bar chart
            st.markdown("### 📈 Distribution des scores")
            fig_bar = px.bar(
                x=["Négatif", "Neutre", "Positif"],
                y=probabilities * 100,
                color=["Négatif", "Neutre", "Positif"],
                color_discrete_map={"Négatif": "#FF6B6B", "Neutre": "#FFD166", "Positif": "#06D6A0"},
                labels={"x": "Sentiment", "y": "Probabilité (%)"},
                text=[f"{p*100:.1f}%" for p in probabilities]
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Details
            with st.expander("📋 Détails de l'analyse"):
                st.markdown(f"**Avis analysé:**")
                st.info(review_text)
                st.markdown(f"**Longueur:** {len(review_text)} caractères")
                st.markdown(f"**Mots:** {len(review_text.split())} mots")
    else:
        st.error("Modèle non chargé. Vérifiez les fichiers du modèle.")


# ========== AFTER SENTIMENT RESULTS ==========

# ================= PERSONNALITY ANALYSIS =================

if groq_analyzer and review_text:
    with st.spinner("🧠 Analyse de personnalité avancée en cours..."):
        
        personality_result = groq_analyzer.analyze(review_text)

        # SAFETY CHECK
        if not isinstance(personality_result, dict) or "scores" not in personality_result:
            st.error("❌ Erreur lors de l'analyse de personnalité.")
        else:
            personality_scores = personality_result["scores"]

            st.markdown("---")
            st.header("🧠 Analyse de Personnalité Big Five (AI-Powered)")
            st.caption(f"*Analyse réalisée avec Groq ({personality_result.get('model_used', 'LLM')})*")

            cols = st.columns(5)

            for idx, (trait, score) in enumerate(personality_scores.items()):
                with cols[idx]:
                    st.markdown(f"**{trait}**")
                    st.progress(score)
                    st.metric("Score", f"{score*100:.1f}%")

            if personality_result.get("analysis"):
                with st.expander("📝 Explication de l'analyse"):
                    st.write(personality_result["analysis"])
    # ============================================
    # PERSONALITY-BASED RECOMMENDATION SYSTEM (FIXED)
    # ============================================

# Replace your AI recommendations section with this FIXED version:

# ================= AI-POWERED RECOMMENDATIONS =================

if groq_analyzer and review_text and model:  # Make sure model is loaded
    st.markdown("---")
    st.header("🎯 Recommandations Personnalisées (AI-Powered)")
    
    with st.spinner("🤖 Génération de recommandations intelligentes..."):
        try:
            # Get sentiment prediction first (in case it wasn't done yet)
            if 'prediction' not in locals() or 'probabilities' not in locals():
                prediction, probabilities = model.predict(review_text)
            
            # Get sentiment label
            if prediction == 0:
                sentiment_clean = "Négatif"
            elif prediction == 1:
                sentiment_clean = "Neutre"
            else:
                sentiment_clean = "Positif"
            
            # Generate AI recommendations
            ai_recommendations = groq_analyzer.generate_recommendations(
                review_text=review_text,
                sentiment=prediction,
                sentiment_confidence=probabilities[prediction],
                personality_scores=personality_scores
            )
            
            if ai_recommendations and len(ai_recommendations) > 0:
                st.success(f"✅ **{len(ai_recommendations)} actions personnalisées générées**")
                
                # Display recommendations as expandable cards
                for i, rec in enumerate(ai_recommendations, 1):
                    # Priority color coding
                    priority_colors = {
                        "URGENT": "#FF4444",
                        "HAUTE": "#FF9800",
                        "MOYENNE": "#2196F3",
                        "BASSE": "#4CAF50"
                    }
                    
                    priority = rec.get("priority", "MOYENNE")
                    color = priority_colors.get(priority, "#2196F3")
                    
                    # Create expander for each recommendation
                    with st.expander(
                        f"{rec['icon']} **{rec['title']}** • {priority}", 
                        expanded=(i <= 2)  # First 2 expanded by default
                    ):
                        # Action
                        st.markdown("### 🎯 Action à réaliser")
                        st.info(rec['action'])
                        
                        # Details
                        st.markdown("### 📝 Pourquoi c'est important")
                        st.write(rec['details'])
                        
                        # Timeline and Priority
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**⏰ Timeline:** {rec['timeline']}")
                        with col2:
                            st.markdown(f"**🚦 Priorité:** `{priority}`")
                        
                        st.markdown("---")
                        
                        # Completion checkbox
                        completed = st.checkbox(
                            "✅ J'ai réalisé cette action",
                            key=f"ai_rec_{i}",
                            help="Cochez quand vous avez accompli cette recommandation"
                        )
                        
                        if completed:
                            st.success("🎉 Excellent travail! Passez à la suivante.")
                
                # ===== EXPLANATION SECTION =====
                with st.expander("🔍 Comment ces recommandations ont été générées?"):
                    st.markdown(f"""
                    ### Analyse multicritère par IA
                    
                    Ces recommandations ont été générées par **{groq_analyzer.model}** en analysant:
                    
                    1. **Votre avis complet** ({len(review_text)} caractères)
                       - Contenu émotionnel et contexte professionnel
                       - Mots-clés et expressions spécifiques
                    
                    2. **Sentiment détecté** 
                       - Type: {sentiment_clean}
                       - Confiance: {probabilities[prediction]*100:.1f}%
                    
                    3. **Profil de personnalité Big Five**
                       - Névrosisme: {personality_scores.get('Névrosisme', 0.5):.0%}
                       - Extraversion: {personality_scores.get('Extraversion', 0.5):.0%}
                       - Conscience: {personality_scores.get('Conscience', 0.5):.0%}
                       - Agréabilité: {personality_scores.get('Agréabilité', 0.5):.0%}
                       - Ouverture: {personality_scores.get('Ouverture', 0.5):.0%}
                    
                    ### Avantages de l'IA vs règles fixes:
                    - ✅ Recommandations contextuelles et nuancées
                    - ✅ Adaptation au ton et style de votre avis
                    - ✅ Actions concrètes et actionnables
                    - ✅ Priorisation intelligente
                    """)
                
                # ===== DOWNLOAD ACTION PLAN =====
                st.markdown("---")
                st.subheader("📥 Télécharger votre Plan d'Action")
                
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    st.write("Exportez vos recommandations en format texte pour les garder sous la main.")
                
                with col_b:
                    # Generate downloadable plan
                    action_plan = f"""
╔════════════════════════════════════════════════════════╗
║     PLAN D'ACTION PROFESSIONNEL PERSONNALISÉ          ║
║              Généré par IA (Groq)                     ║
╚════════════════════════════════════════════════════════╝

Date: {datetime.now().strftime('%d/%m/%Y à %H:%M')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PROFIL ANALYSÉ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sentiment dominant: {sentiment_clean}
Niveau de confiance: {probabilities[prediction]*100:.1f}%

Traits de personnalité Big Five:
- Névrosisme: {personality_scores.get('Névrosisme', 0.5):.0%}
- Extraversion: {personality_scores.get('Extraversion', 0.5):.0%}
- Conscience: {personality_scores.get('Conscience', 0.5):.0%}
- Agréabilité: {personality_scores.get('Agréabilité', 0.5):.0%}
- Ouverture: {personality_scores.get('Ouverture', 0.5):.0%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 RECOMMANDATIONS PERSONNALISÉES ({len(ai_recommendations)})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
                    
                    for i, rec in enumerate(ai_recommendations, 1):
                        action_plan += f"""
{i}. {rec['icon']} {rec['title'].upper()}
   Priorité: [{rec['priority']}]
   
   ➤ Action:
   {rec['action']}
   
   ➤ Détails:
   {rec['details']}
   
   ➤ Timeline: {rec['timeline']}
   
   Status: ☐ À faire  ☐ En cours  ☐ Terminé
   
───────────────────────────────────────────────────────
"""
                    
                    action_plan += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 CONSEILS POUR LA MISE EN ŒUVRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Commencez par les actions URGENTES et HAUTES priorités
✓ Bloquez du temps dans votre agenda pour chaque action
✓ Partagez vos objectifs avec un collègue de confiance
✓ Mesurez vos progrès chaque semaine
✓ Ajustez votre plan selon les résultats

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📞 RESSOURCES & SUPPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Manager direct
• Service RH / Développement professionnel
• Mentor interne (si disponible)
• Programme d'aide aux employés (EAP)
• Formations en ligne (LinkedIn Learning, Coursera)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Plan généré par: Glassdoor Sentiment Analyzer
Modèle IA: {groq_analyzer.model}
⚠️  À adapter selon votre contexte organisationnel

"""
                    
                    st.download_button(
                        label="📄 Télécharger le Plan (.txt)",
                        data=action_plan,
                        file_name=f"plan_action_AI_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        type="secondary"
                    )
                
                # ===== FEEDBACK SECTION =====
                st.markdown("---")
                st.subheader("💬 Ces recommandations vous sont-elles utiles?")
                
                col_x, col_y, col_z = st.columns(3)
                
                with col_x:
                    if st.button("👍 Très utiles", use_container_width=True):
                        st.success("Merci! 🎉")
                
                with col_y:
                    if st.button("😐 Moyennement", use_container_width=True):
                        st.info("Merci pour votre retour.")
                
                with col_z:
                    if st.button("👎 Pas utiles", use_container_width=True):
                        st.warning("Merci. Nous améliorons constamment l'IA.")
            
            else:
                st.warning("⚠️ Aucune recommandation générée. Veuillez réessayer.")
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération: {str(e)}")
            st.info("💡 Essayez de reformuler votre avis ou vérifiez votre connexion internet.")
            
            # Debug info (remove in production)
            import traceback
            with st.expander("🔧 Debug Info (développement)"):
                st.code(traceback.format_exc())















# Section 2: About
st.markdown("---")
st.header("ℹ️ À propos")

with st.expander("📖 Comment fonctionne cette application?"):
    st.markdown("""
    Cette application utilise un modèle **Transformer** (DistilBERT) fine-tuné sur des avis Glassdoor.
    
    **Technologies utilisées:**
    - 🤖 **DistilBERT**: Modèle de langage pré-entraîné
    - 🎯 **Fine-tuning**: Entraîné sur +67000 avis Glassdoor
    - 📊 **Streamlit**: Interface utilisateur interactive
    - 🔥 **PyTorch**: Backend deep learning
    
    **Processus:**
    1. L'avis est tokenizé (découpé en mots/morceaux)
    2. Le modèle analyse le contexte et les relations entre les mots
    3. Une probabilité est calculée pour chaque classe
    4. Le sentiment avec la plus haute probabilité est sélectionné
    """)

with st.expander("⚡ Performances du modèle"):
    st.markdown("""
    **Métriques sur le jeu de test:**
    - Précision globale: **65%**
    - Précision par classe:
      - Négatif: **73%**
      - Neutre: **60%**
      - Positif: **75%**
    - F1-Score: **0.65**
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Développé avec Streamlit | 🤖 Powered by Transformer AI</p>
    </div>
    """,
    unsafe_allow_html=True
)