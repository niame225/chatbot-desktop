from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import logging

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Hugging Face
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_API_KEY')

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
    "Qui t'a conçu": "Raphaël Niamé (+225) 05 06 53 15 22",
    "Qui t'a concu": "Raphaël Niamé (+225) 05 06 53 15 22",
    "Qui t'a concu": "Raphaël Niamé (+225) 05 06 53 15 22",
    "qui est Messy charles": "C'est le père de Manou et Messy",
    "qui t'a conçu": "Raphaël Niamé (+225) 05 06 53 15 22",
    "qui est Awainou Messy charles": "C'est le père de Manou et Messy",
    "Messy charles": "C'est le père de Manou et Messy",
    "Awainou Messy charles": "C'est le père de Manou et Messy",
    "bonsoir": "Bonsoir ! Comment puis-je vous aider ce soir ?",
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

def get_huggingface_response(message):
    """Appel à l'API Hugging Face avec gestion d'erreur complète"""
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        if HUGGINGFACE_TOKEN:
            headers["Authorization"] = f"Bearer {HUGGINGFACE_TOKEN}"
        
        payload = {
            "inputs": message,
            "parameters": {
                "max_length": 50,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": 50256
            },
            "options": {
                "wait_for_model": True
            }
        }
        
        logger.info(f"Envoi requête HF pour: {message}")
        
        response = requests.post(
            HUGGINGFACE_API_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        logger.info(f"Status HF: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                if message in generated_text:
                    generated_text = generated_text.replace(message, '').strip()
                
                if generated_text and len(generated_text) > 3:
                    sentences = generated_text.split('.')
                    clean_response = sentences[0].strip()
                    return clean_response + "." if clean_response else "Je ne sais pas quoi répondre."
                
        elif response.status_code == 402:
            return "⚠️ Service IA temporairement indisponible (limite atteinte). Essayez 'bonjour', 'merci', etc."
        elif response.status_code == 503:
            return "🔄 Modèle en cours de chargement, réessayez dans 30 secondes."
        else:
            logger.error(f"Erreur HF: {response.status_code}")
            return f"❌ Erreur technique (Code: {response.status_code})"
            
    except Exception as e:
        logger.error(f"Exception HF: {str(e)}")
        return "🔧 Erreur technique, réessayez."

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
    
    # 2. API Hugging Face
    return get_huggingface_response(message)

@app.route('/')
def home():
    """Page d'accueil"""
    return render_template('index.html')

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
        'huggingface_token': 'Configuré' if HUGGINGFACE_TOKEN else 'Non configuré ⚠️',
        'routes': ['/chat', '/ask', '/health', '/test', '/debug']
    })

@app.route('/test')
def test():
    """Test de l'API"""
    test_response = get_bot_response("bonjour")
    return jsonify({
        'test_message': 'bonjour',
        'response': test_response,
        'token_status': bool(HUGGINGFACE_TOKEN)
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
        'variables_env': {
            'PORT': os.environ.get('PORT', 'Non défini'),
            'HUGGINGFACE_API_KEY': 'Configuré' if HUGGINGFACE_TOKEN else 'Non configuré'
        },
        'custom_responses_count': len(CUSTOM_RESPONSES)
    })

# Gestion d'erreurs
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Route non trouvée'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Erreur serveur interne'}), 500

if __name__ == '__main__':
    if not HUGGINGFACE_TOKEN:
        logger.warning("⚠️ HUGGINGFACE_API_KEY non configuré")
    else:
        logger.info("✅ Token Hugging Face OK")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Démarrage sur port {port}")
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False
    )