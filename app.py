from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle Qwen localement
try:
    qwen_pipeline = pipeline(
        "text-generation",
        model="Qwen/Qwen2-0.5B-Instruct",
        max_new_tokens=100,
        pad_token_id=50256
    )
    logger.info("✅ Modèle Qwen chargé")
except Exception as e:
    logger.error(f"❌ Échec du chargement du modèle : {str(e)}")
    qwen_pipeline = None

# Réponses personnalisées (fallback)
CUSTOM_RESPONSES = {
    "bonjour": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "salut": "Salut ! Que puis-je faire pour vous ?",
    "comment ça va": "Je vais bien, merci ! Et vous ?",
    "au revoir": "Au revoir ! Passez une excellente journée !",
    "merci": "De rien, je suis là pour vous aider !",
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I do for you?",
    "qui t'a conçu": "Raphaël Niamé (+225) 05 06 53 15 22",
    "qui t'a concu": "Raphaël Niamé (+225) 05 06 53 15 22",
    "qui t'a fait": "Raphaël Niamé (+225) 05 06 53 15 22",
    "qui est messy charles": "C'est le père de Manou et Messy",
    "bonsoir": "Bonsoir ! Comment puis-je vous aider ce soir ?",
    "bien": "OK. Que puis-je faire pour vous ?",
    "très bien": "OK. Que puis-je faire pour vous ?",
    "je vais bien et toi": "Très bien. Comment puis-je vous aider ?",
    "comment vas-tu": "Je vais bien, merci ! Et vous ?",
    "comment allez-vous": "Je vais bien, merci ! Et vous ?",
    "comment t'appelles-tu": "Je suis un assistant virtuel sans nom propre. Tu peux m'appeler simplement 'Assistant'. Comment puis-je t'aider aujourd'hui ?",
    "qui es-tu": "Je suis un assistant virtuel sans nom propre. Tu peux m'appeler simplement 'Assistant'. Comment puis-je t'aider aujourd'hui ?",
    "quel est ton nom": "Je suis un assistant virtuel sans nom propre. Tu peux m'appeler simplement 'Assistant'. Comment puis-je t'aider aujourd'hui ?",
    "tu t'appelles comment": "Je suis un assistant virtuel sans nom propre. Tu peux m'appeler simplement 'Assistant'. Comment puis-je t'aider aujourd'hui ?",
    "qui est raphael niame": "C'est un développeur freelance d'applications web, mobiles et bureaux.",
    "comment contacter raphael niame": "Raphaël Niamé: (+225) 05 06 53 15 22",
    "email": "niame225@gmail.com",
    "oulai": "c'est le père de Tchounatchou",
    "soma": "c'est un célèbre agent immobilier",
    "oré Roland": "C'est un grand homme d'affaires. Il est ivoirien"
}

def get_custom_response(message):
    """Cherche une réponse personnalisée avec correspondance de mots-clés multiples"""
    if not message:
        return None
        
    message_lower = message.lower().strip()
    # Nettoyer le message : supprimer la ponctuation et séparer en mots
    import re
    # Remplacer la ponctuation par des espaces
    clean_message = re.sub(r'[^\w\s]', ' ', message_lower)
    message_words = set(clean_message.split())  # Utiliser un set pour une recherche plus rapide
    
    # Parcourir toutes les réponses personnalisées
    for key, response in CUSTOM_RESPONSES.items():
        # Nettoyer la clé de la même façon
        clean_key = re.sub(r'[^\w\s]', ' ', key.lower())
        key_words = clean_key.split()
        
        # Vérifier si tous les mots de la clé sont dans le message
        if key_words and all(word in message_words for word in key_words):
            logger.info(f"Correspondance trouvée: '{key}' -> '{message}'")
            return response
    
    # Si aucune correspondance exacte, essayer une correspondance partielle
    for key, response in CUSTOM_RESPONSES.items():
        key_lower = key.lower()
        if key_lower in message_lower or message_lower in key_lower:
            logger.info(f"Correspondance partielle: '{key}' -> '{message}'")
            return response
    
    return None

def get_local_model_response(message):
    """Utilise un modèle local (Qwen2-0.5B) pour générer une réponse"""
    try:
        logger.info(f"Envoi au modèle local: {message}")

        prompt = f"User: {message}\nAssistant:"
        response = qwen_pipeline(prompt)

        if response and len(response) > 0:
            generated_text = response[0]['generated_text'].replace(prompt, '').strip()

            if generated_text:
                return generated_text.split('\n')[0][:250]  # Limite à 250 caractères

        return "Je n'ai pas réussi à comprendre votre question."

    except Exception as e:
        logger.error(f"Erreur modèle local: {str(e)}")
        return "Erreur lors de la génération de la réponse."

def get_bot_response(message):
    """Fonction principale pour obtenir une réponse"""
    if not message or len(message.strip()) == 0:
        return "Veuillez saisir un message."

    if len(message) > 200:
        return "Message trop long (max 200 caractères)."

    # 1. Réponses personnalisées d'abord
    custom_response = get_custom_response(message)
    if custom_response:
        logger.info(f"Réponse personnalisée pour: {message}")
        return custom_response

    # 2. Modèle local IA
    if qwen_pipeline:
        return get_local_model_response(message)
    else:
        return "⚠️ Le modèle IA est temporairement indisponible."

@app.route('/')
def home():
    """Page d'accueil basique"""
    return jsonify({
        "status": "Chatbot prêt",
        "description": "Envoyez une requête POST à /chat pour commencer.",
        "routes": ["/chat", "/ask", "/health", "/test", "/debug"]
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Route principale pour le chat (JSON)"""
    try:
        logger.info("Requête reçue sur /chat")

        # Vérifier le content-type
        if not request.is_json:
            logger.error("Content-Type n'est pas JSON")
            return jsonify({'error': 'Content-Type doit être application/json'}), 400

        data = request.get_json()
        if not data:
            logger.error("Pas de données JSON")
            return jsonify({'error': 'Données JSON manquantes'}), 400

        user_message = data.get('message', '').strip()
        logger.info(f"Message reçu: '{user_message}'")

        if not user_message:
            return jsonify({'response': 'Veuillez saisir un message.'})

        bot_response = get_bot_response(user_message)
        logger.info(f"Réponse envoyée: '{bot_response}'")

        return jsonify({'response': bot_response})

    except Exception as e:
        logger.error(f"Erreur dans /chat: {str(e)}")
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Route pour compatibilité (FormData et JSON)"""
    try:
        logger.info("Requête reçue sur /ask")

        # Support FormData ET JSON
        if request.is_json:
            data = request.get_json()
            user_message = data.get('message', '').strip()
        else:
            user_message = request.form.get('message', '').strip()

        logger.info(f"Message /ask: '{user_message}'")

        if not user_message:
            return jsonify({'response': 'Veuillez saisir un message.'})

        bot_response = get_bot_response(user_message)
        return jsonify({'response': bot_response})

    except Exception as e:
        logger.error(f"Erreur /ask: {str(e)}")
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

@app.route('/health')
def health():
    """Vérification santé"""
    return jsonify({
        'status': 'OK ✅',
        'message': 'Flask fonctionne',
        'model_status': 'Actif ✅' if qwen_pipeline else 'Inactif ⚠️',
        'routes': ['/chat', '/ask', '/health', '/test', '/debug']
    })

@app.route('/test')
def test():
    """Test de l'API"""
    test_response = get_bot_response("bonjour")
    return jsonify({
        'test_message': 'bonjour',
        'response': test_response,
        'model_status': bool(qwen_pipeline)
    })

@app.route('/debug')
def debug():
    """Debug complet"""
    return jsonify({
        'flask_status': 'Actif ✅',
        'routes_disponibles': [
            'GET /',
            'POST /chat (JSON)',
            'POST /ask (FormData/JSON)', 
            'GET /health',
            'GET /test',
            'GET /debug'
        ],
        'custom_responses_count': len(CUSTOM_RESPONSES),
        'model_loaded': bool(qwen_pipeline)
    })

# Gestion d'erreurs
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Route non trouvée'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Erreur serveur interne'}), 500

if __name__ == '__main__':
    logger.info("🚀 Démarrage du chatbot avec modèle local")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)