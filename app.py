from flask import Flask, request, jsonify, render_template, redirect
from huggingface_hub import InferenceClient
from requests.exceptions import RequestException
import datetime
import os
import re
import unicodedata
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialisation du client Hugging Face
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=api_key
)

# Créer le dossier logs si nécessaire
os.makedirs("logs", exist_ok=True)

# Nom du fichier log
now = datetime.datetime.now()
filename = now.strftime("logs/Chat_%Y-%m-%d_%H-%M.txt")

# Fonction de log
def log_conversation(line):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Réponses personnalisées
custom_responses = {
    "bonjour": "Bonjour ! Je suis là pour t'aider. Comment puis-je t'assister aujourd'hui ?",
    "qui t'a conçu": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "comment vas tu": "Je vais bien, merci pour ton intérêt. Comment ça va pour toi ?",
    "qui t'a fait": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "qui t'a créé": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "qui est raphael niamé": "C'est un développeur freelance d'applications.",
    "raphael niamé": "C'est un développeur freelance d'applications.",
    "qui est oulai": "C'est le père de Tchounatchou ou Djouniédjou.",
    "oulai": "C'est le père de Tchounatchou ou Djouniédjou.",
    "qui est diarrassouba soma": "C'est un célèbre agent immobilier.",
    "qui soma": "C'est un célèbre agent immobilier.",
    "diarrassouba": "C'est un célèbre agent immobilier."
}

# Fonction de normalisation
def normalize(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r'[^\w\s]', '', text).strip().lower().replace("  ", " ")

# Application Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Aucune question envoyée."})

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
        generated = ""
        for message in client.text_generation(
            prompt=f"<s>[INST] Réponds brièvement en français à la question suivante : {user_input} [/INST]",
            max_new_tokens=150,
            stream=True,
            temperature=0.5
        ):
            content = message or ""
            generated += content

        log_conversation(f"Vous: {user_input}\nAssistant: {generated}")
        return jsonify({"response": generated})

    except RequestException as e:
        error = f"Erreur réseau : {str(e)}"
        log_conversation(f"Vous: {user_input}\nAssistant: {error}")
        return jsonify({"error": error})

    except Exception as e:
        error = f"Erreur serveur : {str(e)}"
        log_conversation(f"Vous: {user_input}\nAssistant: {error}")
        return jsonify({"error": error})

if __name__ == "__main__":
    app.run(port=10000)