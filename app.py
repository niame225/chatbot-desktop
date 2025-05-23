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
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HUGGINGFACE_API_KEY)

# Réponses personnalisées
custom_responses = {
    "bonjour": "Bonjour ! Comment puis-je vous aider ?",
    "salut": "Bonjour ! Comment puis-je vous aider ?",
    "hello": "Bonjour ! Comment puis-je vous aider ?",
    "comment ça va": "Je vais bien, merci ! Et vous ?",
    "qui es-tu": "Je suis un assistant IA basé sur le modèle Zephyr-7B.",
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

    # Sinon, appeler Hugging Face
    try:
        payload = {
            "inputs": f"[INST] Réponds en français : {user_input} [/INST]",
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "return_full_text": False
            }
        }

        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta ",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        generated_text = data[0].get("generated_text", "") if isinstance(data, list) else str(data)
        cleaned = generated_text.split("[/INST]")[-1].strip() if "[/INST]" in generated_text else generated_text.strip()

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