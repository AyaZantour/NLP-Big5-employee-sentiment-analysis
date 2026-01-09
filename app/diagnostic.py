# diagnostic.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🔍 Diagnostic du modèle fine-tuned")
print("=" * 60)

# ============ CORRECTION DES CHEMINS ============
# Va un dossier en arrière (..) pour trouver models/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'transformers_model')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'transformers_tokenizer')

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Model path: {MODEL_PATH}")
print(f"📁 Tokenizer path: {TOKENIZER_PATH}")

# Vérifie que les fichiers existent
print("\n🔍 Vérification des fichiers:")
for path, name in [(MODEL_PATH, "Modèle"), (TOKENIZER_PATH, "Tokenizer")]:
    if os.path.exists(path):
        print(f"✅ {name}: {path}")
        # Liste les fichiers
        files = os.listdir(path)
        print(f"   Fichiers: {len(files)} fichiers")
        for f in files[:3]:  # Montre les 3 premiers
            print(f"   - {f}")
    else:
        print(f"❌ {name}: NON TROUVÉ à {path}")

# Charge seulement si les fichiers existent
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        
        print(f"\n✅ Modèle chargé!")
        print(f"📋 Labels configurés: {model.config.id2label}")
        
        # Test avec des phrases TRÈS claires
        test_cases = [
            ("everything is perfect i love it here", "should be POSITIF"),
            ("its horrible,  toxic place", "Devrait être NÉGATIF"),
            ("nothing special", "Devrait être NEUTRE"),
            ("Je veux démissionner tellement c'est mauvais", "Devrait être NÉGATIF"),
            ("Meilleure entreprise de ma vie", "Devrait être POSITIF"),
        ]
        
        print("\n🧪 Tests:")
        for text, expected in test_cases:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0]
                prediction = torch.argmax(outputs.logits, dim=1).item()
            
            predicted_label = model.config.id2label[prediction]
            
            print(f"\n📝 '{text[:50]}...'")
            print(f"   Attendu: {expected}")
            print(f"   Prédit: {predicted_label}")
            print(f"   Confiance: N={probabilities[0]:.1%}, Neu={probabilities[1]:.1%}, P={probabilities[2]:.1%}")
            
            # Analyse
            if "POSITIF" in expected and "Negative" in predicted_label:
                print("   ⚠️  PROBLÈME: Inverse!")
            elif "NÉGATIF" in expected and "Positive" in predicted_label:
                print("   ⚠️  PROBLÈME: Inverse!")
                
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
else:
    print("\n❌ Fichiers manquants. Structure actuelle:")
    base = os.path.dirname(BASE_DIR)
    for root, dirs, files in os.walk(base):
        level = root.replace(base, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}📂 {os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:
            if file.endswith(('.json', '.pkl', '.bin', '.safetensors')):
                print(f'{subindent}📄 {file}')