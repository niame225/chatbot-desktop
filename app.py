from flask import Flask, request, jsonify, render_template
import requests
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from requests.exceptions import RequestException

# Charger les variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ]
)

# Initialiser Flask
app = Flask(__name__)

# Charger la clé API HuggingFace
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    logging.error("La clé API HUGGINGFACE_API_KEY n'est pas définie.")
    raise ValueError("La clé API HUGGINGFACE_API_KEY n'est pas définie.")

# Initialiser le client HuggingFace
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(model=MODEL_NAME, token=HUGGINGFACE_API_KEY)

# Réponses personnalisées (correspondance exacte uniquement)
custom_responses = {
    "bonjour": "Bonjour ! Comment puis-je vous aider ?",
    "salut": "Salut ! Comment puis-je vous aider ?",
    "hello": "Salut ! Comment puis-je vous aider ?",
    "comment ça va": "Je vais bien, merci ! Et vous ?",
    "comment ça va ?": "Je vais bien, merci ! Et vous ?",
    "qui es-tu": "Je suis un assistant IA basé sur le modèle GPT.",
    "qui es-tu ?": "Je suis un assistant IA basé sur le modèle GPT.",
    "qui t'a conçu": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "qui t'a conçu ?": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "soma": "C'est un célèbre agent immobilier.",
    "oulai": "C'est le père de Tchounatchou.",
    "messy charles": "C'est le père de Manou.",
    "qui est raphaël niamé": "Raphaël Niamé est un développeur freelance d'applications web, mobiles et bureau.",
    "qui est raphaël niamé ?": "Raphaël Niamé est un développeur freelance d'applications web, mobiles et bureau.",
}

def normalize(text):
    return text.lower().strip()

def log_conversation(entry):
    try:
        with open("conversation.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {entry}\n")
    except Exception as e:
        logging.error(f"Erreur lors de l'écriture du log : {e}")

def get_custom_response(user_input):
    """Vérifier uniquement les correspondances exactes pour les réponses personnalisées"""
    normalized_input = normalize(user_input)
    
    # Correspondance exacte d'abord
    if normalized_input in custom_responses:
        return custom_responses[normalized_input]
    
    # Quelques correspondances spéciales pour des variations courantes
    special_cases = {
        "salutation": ["bonjour", "salut", "hello", "hi", "bonsoir"],
        "état": ["comment ça va", "comment allez-vous", "ça va"],
        "identité": ["qui es-tu", "que fais-tu", "ton nom"]
    }
    
    for category, keywords in special_cases.items():
        for keyword in keywords:
            if normalized_input == keyword or normalized_input == keyword + "?":
                if keyword in custom_responses:
                    return custom_responses[keyword]
    
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Aucune question envoyée."}), 400

    logging.info(f"Question reçue: {user_input}")

    # Vérifier les réponses personnalisées (correspondance exacte uniquement)
    custom_response = get_custom_response(user_input)
    
    if custom_response:
        logging.info(f"Réponse personnalisée trouvée: {custom_response}")
        log_conversation(f"Vous: {user_input}\nAssistant: {custom_response}")
        return jsonify({"response": custom_response})

    # Sinon, appeler Hugging Face avec InferenceClient
    logging.info("Appel à l'API Hugging Face...")
    try:
        prompt = f"[INST] Réponds en français de manière concise et utile : {user_input} [/INST]"
        response = ""

        for token in client.text_generation(
            prompt=prompt, 
            max_new_tokens=150, 
            stream=True,
            temperature=0.7,
            do_sample=True
        ):
            response += token or ""

        cleaned = response.strip()
        if not cleaned:
            cleaned = "Je n'ai pas pu générer une réponse appropriée."

        logging.info(f"Réponse IA générée: {cleaned[:100]}...")
        log_conversation(f"Vous: {user_input}\nAssistant: {cleaned}")
        return jsonify({"response": cleaned})

    except RequestException as e:
        error = f"Erreur réseau avec Hugging Face : {str(e)}"
        logging.error(error)
        log_conversation(f"Vous: {user_input}\nAssistant: {error}")
        return jsonify({"error": error}), 503

    except Exception as e:
        error = f"Erreur serveur : {str(e)}"
        logging.error(error)
        log_conversation(f"Vous: {user_input}\nAssistant: {error}")
        return jsonify({"error": error}), 500

@app.route("/test")
def test():
    return jsonify({
        "status": "OK",
        "api_key_configured": bool(HUGGINGFACE_API_KEY),
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/debug", methods=["POST"])
def debug():
    """Route de debug pour tester l'API Hugging Face directement"""
    try:
        test_prompt = "[INST] Bonjour, comment allez-vous ? [/INST]"
        response = ""
        
        for token in client.text_generation(prompt=test_prompt, max_new_tokens=50, stream=True):
            response += token or ""
            
        return jsonify({
            "status": "success",
            "response": response.strip(),
            "model": MODEL_NAME
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Démarrage de l'application sur le port {port}")
    logging.info(f"Modèle utilisé: {MODEL_NAME}")
    app.run(host="0.0.0.0", port=port, debug=False)