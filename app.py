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

# Réponses personnalisées
custom_responses = {
    "bonjour": "Bonjour ! Comment puis-je vous aider ?",
    "salut": "Salut ! Comment puis-je vous aider ?",
    "hello": "Salut ! Comment puis-je vous aider ?",
    "comment ça va": "Je vais bien, merci ! Et vous ?",
    "qui es-tu": "Je suis un assistant IA basé sur le modèle GPT.",
    "qui t'a conçu": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "Soma": "C'est un célèbre agent immobilier.",
    "oulai": "C'est le père de Tchounatchou.",
    "Messy Charles": "C'est le père de Manou.",
    "qui est Raphaël Niamé ": "Raphaël Niamé est un développeur freelance d'applications web,  mobiles et bureau.",
}

def normalize(text):
    return text.lower().strip()

def log_conversation(entry):
    try:
        with open("conversation.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {entry}\n")
    except Exception as e:
        logging.error(f"Erreur lors de l'écriture du log : {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Aucune question envoyée."}), 400

    normalized_input = normalize(user_input)
    matched_response = None

    # Vérifier les réponses personnalisées
    for key in custom_responses:
        if normalize(key) in normalized_input:
            matched_response = custom_responses[key]
            break

    if matched_response:
        log_conversation(f"Vous: {user_input}\nAssistant: {matched_response}")
        return jsonify({"response": matched_response})

    # Sinon, appeler Hugging Face avec InferenceClient
    try:
        prompt = f"[INST] Réponds en français : {user_input} [/INST]"
        response = ""

        for token in client.text_generation(prompt=prompt, max_new_tokens=150, stream=True):
            response += token or ""

        cleaned = response.strip()
        if not cleaned:
            cleaned = "Je n'ai pas pu générer une réponse appropriée."

        log_conversation(f"Vous: {user_input}\nAssistant: {cleaned}")
        return jsonify({"response": cleaned})

    except RequestException as e:
        error = f"Erreur réseau : {str(e)}"
        log_conversation(f"Vous: {user_input}\nAssistant: {error}")
        return jsonify({"error": error}), 503

    except Exception as e:
        error = f"Erreur serveur : {str(e)}"
        log_conversation(f"Vous: {user_input}\nAssistant: {error}")
        return jsonify({"error": error}), 500
    except RequestException as e:
        error = f"Erreur réseau : {str(e)}"
        log_conversation(f"Vous: {user_input}\nAssistant: {error}")
        return jsonify({"error": error}), 503

    except Exception as e:
        error = f"Erreur serveur : {str(e)}"
        log_conversation(f"Vous: {user_input}\nAssistant: {error}")
        return jsonify({"error": error}), 500

@app.route("/test")
def test():
    return jsonify({
        "status": "OK",
        "api_key_configured": bool(HUGGINGFACE_API_KEY),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Démarrage de l'application sur le port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)