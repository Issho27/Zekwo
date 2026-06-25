from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import ollama
import json
import os
import uuid
import sqlite3
from datetime import datetime
from duckduckgo_search import DDGS
import requests
import random
import feedparser
import chromadb
import httpx
import threading
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from werkzeug.security import generate_password_hash, check_password_hash

# ============== CONFIG ==============

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "static", "chroma_db")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
WEATHER_API_KEY = os.getenv("RAPIDAPI_KEY")

DB_PATH = os.path.join(os.path.dirname(__file__), "static", "users.db")

app = Flask(__name__)
app.secret_key = "zekwo-v2-secret-key-change-me"

MODEL_ACTIF = "qwen3:8b"

MODELES_OLLAMA = {"qwythos-9b", "qwen3:8b", "gemma4-agentic", "phi4-mini"}

MODELES_DISPONIBLES = {
    "qwythos-9b":      "Qwythos 9B (local)",
    "qwen3:8b":        "Qwen 3 8B (local)",
    "gemma4-agentic":  "Gemma 4 Agentic (local)",
    "phi4-mini":       "Phi 4 Mini (local)",
}

MODELES_AVEC_THINKING = {"qwen3:8b", "qwythos-9b", "gemma4-agentic"}
MODELES_SANS_TOOLS = {"phi4-mini"}
MODELES_SANS_WEB = {"qwythos-9b"}
MODELES_UNCENSORED = {"qwythos-9b"}

SPORT_HOST = "sportapi7.p.rapidapi.com"
SPORTS_DISPONIBLES = [
    "football", "basketball", "tennis", "american-football", "baseball",
    "ice-hockey", "rugby", "volleyball", "cricket", "motorsport"
]
CATEGORIES_RSS = ["general", "sport", "tech", "economie", "sciences", "international"]

# ============== BASE DE DONNÉES UTILISATEURS ==============

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

def create_user(username, password):
    user_id = str(uuid.uuid4())[:12]
    password_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, password_hash, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

init_db()

# ============== HELPER AUTH ==============

def utilisateur_connecte():
    return session.get("user_id") is not None

def utilisateur_actuel():
    return session.get("user_id")

def nom_utilisateur_actuel():
    return session.get("username", "inconnu")

# ============== OUTILS ==============

_chroma_client = None
_chroma_collection = None

def get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _chroma_collection = _chroma_client.get_collection(name="knowledge", embedding_function=ef)
        return _chroma_collection
    except Exception as e:
        print(f"Base de connaissances non disponible : {e}")
        return None

def recherche_sport(sport, query=""):
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://{SPORT_HOST}/api/v1/sport/{sport}/scheduled-events/{today}"
    headers = {"x-rapidapi-host": SPORT_HOST, "x-rapidapi-key": RAPIDAPI_KEY}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        events = data.get("events", [])
        if not events:
            return f"Aucun événement {sport} trouvé aujourd'hui."
        contexte = f"Voici les événements {sport} du jour ({today}) :\n"
        for e in events[:6]:
            try:
                home = e["homeTeam"]["name"]
                away = e["awayTeam"]["name"]
                tournoi = e["tournament"]["name"]
                statut = e["status"]["description"]
                score_str = ""
                if "homeScore" in e and "current" in e["homeScore"]:
                    score_str = f"{e['homeScore']['current']}-{e['awayScore']['current']}"
                contexte += f"- {tournoi} : {home} {score_str} {away} ({statut})\n"
            except:
                continue
        return contexte
    except Exception as e:
        return f"Erreur lors de la recherche sportive : impossible de récupérer les données."

def recherche_meteo(ville="Paris"):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={ville}&appid={WEATHER_API_KEY}&units=metric&lang=fr"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Impossible de trouver la météo pour '{ville}'."
        temp = data["main"]["temp"]
        ressenti = data["main"]["feels_like"]
        description = data["weather"][0]["description"]
        humidite = data["main"]["humidity"]
        vent = data["wind"]["speed"]
        return f"Météo actuelle à {ville} : {description}, {temp}°C (ressenti {ressenti}°C), humidité {humidite}%, vent {vent} km/h.\n"
    except Exception as e:
        return "Erreur lors de la recherche météo."

def recherche_wikipedia(sujet):
    HEADERS = {"User-Agent": "Zekwo/2.0 (contact@zekwo.local)"}
    try:
        for langue in ["fr", "en"]:
            url_search = "https://fr.wikipedia.org/w/api.php" if langue == "fr" else "https://en.wikipedia.org/w/api.php"
            params_search = {"action": "query", "list": "search", "srsearch": sujet, "format": "json", "srlimit": 1}
            r = requests.get(url_search, params=params_search, timeout=5, headers=HEADERS)
            data = r.json()
            resultats = data.get("query", {}).get("search", [])
            if not resultats:
                continue
            titre = resultats[0]["title"]
            params_extract = {"action": "query", "prop": "extracts", "exintro": True, "explaintext": True, "titles": titre, "format": "json", "exsentences": 4}
            r2 = requests.get(url_search, params=params_extract, timeout=5, headers=HEADERS)
            data2 = r2.json()
            pages = data2.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            extrait = page.get("extract", "").strip()
            if extrait:
                return f"Voici un extrait de l'article Wikipédia \"{titre}\" :\n{extrait}\n"
        return f"Aucun article Wikipédia trouvé pour '{sujet}'."
    except Exception as e:
        return f"Erreur lors de la recherche Wikipédia pour '{sujet}'."

def recherche_web(requete):
    try:
        with DDGS() as ddgs:
            resultats = list(ddgs.text(requete, max_results=5, region="fr_fr", timelimit="w"))
            if not resultats:
                return "Aucun résultat trouvé sur le web."
            contexte = "Voici des informations trouvées sur internet. Base-toi UNIQUEMENT sur ces infos :\n"
            for r in resultats:
                contexte += f"- {r['title']} : {r['body']}\n"
            return contexte
    except Exception as e:
        return "Erreur lors de la recherche web."

FLUX_RSS_FILE = os.path.join(os.path.dirname(__file__), "flux_rss.json")

def charger_flux_rss():
    if os.path.exists(FLUX_RSS_FILE):
        with open(FLUX_RSS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def recherche_actualite(categorie):
    flux = charger_flux_rss()
    urls = flux.get(categorie, [])
    if not urls:
        return f"Aucun flux RSS configuré pour la catégorie '{categorie}'."
    contexte = f"Voici des actualités récentes (catégorie : {categorie}) :\n"
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                contexte += f"- {entry.title} : {entry.get('summary', '')[:200]}\n"
        except:
            continue
    return contexte

def chercher_dans_cours(question):
    collection = get_chroma_collection()
    if not collection:
        return "La base de connaissances n'est pas disponible."
    try:
        resultats = collection.query(query_texts=[question], n_results=4)
        documents = resultats.get("documents", [[]])[0]
        metadatas = resultats.get("metadatas", [[]])[0]
        if not documents:
            return "Aucun passage pertinent trouvé dans la base de connaissances."
        contexte = "Voici des passages pertinents issus des cours :\n\n"
        for doc, meta in zip(documents, metadatas):
            source = meta.get("source", "inconnu")
            contexte += f"[Source : {source}]\n{doc}\n\n"
        return contexte.strip()
    except Exception as e:
        return "Erreur lors de la recherche dans la base de connaissances."

# ============== OUTILS TOOL CALLING ==============

# Noms internes des outils — servent de clés pour le filtrage front→back
OUTILS_PAR_NOM = {
    "recherche_actualite": {
        "type": "function",
        "function": {
            "name": "recherche_actualite",
            "description": "Cherche des actualités récentes dans une catégorie précise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categorie": {"type": "string", "enum": CATEGORIES_RSS}
                },
                "required": ["categorie"]
            }
        }
    },
    "recherche_wikipedia": {
        "type": "function",
        "function": {
            "name": "recherche_wikipedia",
            "description": "Cherche une information factuelle sur Wikipédia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sujet": {"type": "string"}
                },
                "required": ["sujet"]
            }
        }
    },
    "chercher_dans_cours": {
        "type": "function",
        "function": {
            "name": "chercher_dans_cours",
            "description": "Cherche dans la base de connaissances personnelle de l'utilisateur. Contient des documents sur tous les sujets. À utiliser EN PRIORITÉ pour toute question factuelle ou de connaissance, sauf si la météo ou le sport en temps réel sont clairement plus pertinents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            }
        }
    },
    "recherche_meteo": {
        "type": "function",
        "function": {
            "name": "recherche_meteo",
            "description": "Donne la météo actuelle d'une ville.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ville": {"type": "string"}
                },
                "required": ["ville"]
            }
        }
    },
    "recherche_sport": {
        "type": "function",
        "function": {
            "name": "recherche_sport",
            "description": "Donne les résultats ou matchs du jour pour un sport précis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sport": {"type": "string", "enum": SPORTS_DISPONIBLES}
                },
                "required": ["sport"]
            }
        }
    },
    "recherche_web": {
        "type": "function",
        "function": {
            "name": "recherche_web",
            "description": "Recherche web générale pour toute info non couverte par les autres outils.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requete": {"type": "string"}
                },
                "required": ["requete"]
            }
        }
    }
}

OUTILS = list(OUTILS_PAR_NOM.values())

# Outils interdits pour Qwythos (hallucine sans garde-fous)
OUTILS_INTERDITS_QWYTHOS = {"recherche_wikipedia"}

def executer_outil(nom, arguments):
    try:
        if nom == "recherche_actualite":
            return recherche_actualite(arguments.get("categorie", "general"))
        elif nom == "recherche_wikipedia":
            return recherche_wikipedia(arguments.get("sujet", ""))
        elif nom == "recherche_meteo":
            return recherche_meteo(arguments.get("ville", "Paris"))
        elif nom == "recherche_sport":
            return recherche_sport(arguments.get("sport", ""))
        elif nom == "recherche_web":
            return recherche_web(arguments.get("requete", ""))
        elif nom == "chercher_dans_cours":
            return chercher_dans_cours(arguments.get("question", ""))
        return f"Outil inconnu : {nom}"
    except Exception as e:
        return f"Erreur lors de l'exécution de l'outil {nom}."

# ============== CONVERSATIONS ==============

CONV_DIR = os.path.join(os.path.dirname(__file__), "static", "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

PERSONNALITE = """Ton nom est Zekwo. Tu es cool, plus un ami qu'un assistant même si ton but est aussi d'aider les gens. Tu tutoies tes interlocuteurs, sauf contre-indication de leur part. Tu te souviens de tout ce qui est dit dans la conversation. Tu parles dans un langage très courant, pas formel ni institutionnel, tu ne prends pas de haut ton interlocuteur, et très important : tu ne vois pas la conversation comme une chatbox, mais comme un endroit. Tu pourras dire 'Content de te voir ici !' si le contexte le demande, quelque chose qui montre que tu vois cette conversation comme une place virtuelle et pas comme un simple échange. Tu as accès à plusieurs outils (actualités, Wikipédia, météo, sport, recherche web) : utilise-les chaque fois qu'une question porte sur une information factuelle, récente, ou que tu n'es pas sûr de connaître avec certitude, mais utilise toujours la base de donnée en premier, sauf si la question porte sur la météo ou le sport. Quand on te donne des résultats d'outils, tu ne les modifies JAMAIS et tu n'inventes JAMAIS de détails supplémentaires. Si un outil ne renvoie rien d'utile, dis honnêtement que tu ne sais pas plutôt que d'inventer. Tu as accès à une base de connaissances personnelle via l'outil chercher_dans_cours. Elle peut contenir des documents sur n'importe quel sujet. Utilise-la EN PREMIER pour toute question factuelle ou de connaissance, avant de recourir au web ou à Wikipédia. Ne l'utilise pas pour la météo ou les résultats sportifs en temps réel. Sois proactif avec les outils : dès qu'une question implique un fait vérifiable, utilise l'outil correspondant MÊME SI la formulation est familière ou indirecte."""

def charger_conversation(conv_id):
    path = f"{CONV_DIR}/{conv_id}.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

def sauvegarder_conversation(conv):
    path = f"{CONV_DIR}/{conv['id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)

def generer_titre(premier_message):
    prompt = f"Génère un titre très court (5 mots max) pour cette conversation : {premier_message[:200]}"
    if MODEL_ACTIF in MODELES_OLLAMA:
        response = ollama.chat(model=MODEL_ACTIF, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip().splitlines()[0]
    return "Nouvelle conversation"

def generer_titre_async(conv_id):
    conv = charger_conversation(conv_id)
    if not conv:
        return
    messages_user = [m for m in conv["messages"] if m["role"] == "user"]
    if messages_user:
        try:
            conv["titre"] = generer_titre(messages_user[0]["content"])
            sauvegarder_conversation(conv)
        except Exception as e:
            print(f"Erreur génération titre async pour {conv_id} : {e}")

# ============== ROUTES AUTH ==============

@app.route("/")
def index():
    if not utilisateur_connecte():
        return redirect(url_for("login_page"))
    return redirect(url_for("chat_page"))

@app.route("/login")
def login_page():
    if utilisateur_connecte():
        return redirect(url_for("chat_page"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"erreur": "Champs manquants"}), 400
    user = get_user_by_username(username)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"erreur": "Identifiants incorrects"}), 401
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return jsonify({"ok": True})

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"erreur": "Champs manquants"}), 400
    if len(username) < 2:
        return jsonify({"erreur": "Nom trop court (2 caractères min)"}), 400
    if len(password) < 4:
        return jsonify({"erreur": "Mot de passe trop court (4 caractères min)"}), 400
    user_id = create_user(username, password)
    if not user_id:
        return jsonify({"erreur": "Ce nom est déjà pris"}), 409
    session["user_id"] = user_id
    session["username"] = username
    return jsonify({"ok": True})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/me")
def me():
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401
    return jsonify({"username": nom_utilisateur_actuel(), "user_id": utilisateur_actuel()})

# ============== ROUTES PAGES ==============

@app.route("/chat")
def chat_page():
    if not utilisateur_connecte():
        return redirect(url_for("login_page"))
    return render_template("index.html")

@app.route("/novae")
def novae_page():
    if not utilisateur_connecte():
        return redirect(url_for("login_page"))
    return render_template("novae.html")

# ============== FICHE NOVA ==============

SPECTRAL_CLASSES = ["O", "B", "A", "F", "G", "K", "M"]
SPECTRAL_TEMPS = {
    "O": "~30 000 K", "B": "~15 000 K", "A": "~8 500 K",
    "F": "~6 500 K", "G": "~5 800 K", "K": "~4 500 K", "M": "~3 200 K"
}
STATUTS = ["Stable", "En expansion", "Phase critique"]

@app.route("/conversations/<conv_id>/fiche", methods=["GET"])
def fiche_nova(conv_id):
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401
    conv = charger_conversation(conv_id)
    if not conv or conv.get("user_id") != utilisateur_actuel():
        return jsonify({"erreur": "introuvable"}), 404
    seed = sum(ord(c) for c in conv_id)
    rand = random.Random(seed)
    designation = f"NV-{conv['date'][:10].replace('-', '')}-{conv_id[:4].upper()}"
    nb_messages = len([m for m in conv["messages"] if m["role"] in ("user", "assistant")])
    magnitude = round(nb_messages / 2, 1)
    classe = rand.choice(SPECTRAL_CLASSES)
    statut = rand.choice(STATUTS)
    distance = round(rand.uniform(0.5, 850), 1)
    return jsonify({
        "designation": designation,
        "constellation_origine": conv["date"][:10],
        "magnitude": magnitude,
        "classe_spectrale": classe,
        "temperature": SPECTRAL_TEMPS[classe],
        "distance_al": distance,
        "statut": statut
    })

# ============== CONVERSATIONS (NOVAE) ==============

@app.route("/conversations", methods=["GET"])
def lister_conversations():
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401
    user_id = utilisateur_actuel()
    conversations = []
    for fichier in os.listdir(CONV_DIR):
        if fichier.endswith(".json"):
            try:
                with open(f"{CONV_DIR}/{fichier}", "r", encoding="utf-8") as f:
                    conv = json.load(f)
                    if conv.get("user_id") == user_id:
                        conversations.append({
                            "id": conv["id"],
                            "titre": conv["titre"],
                            "date": conv["date"]
                        })
            except:
                continue
    conversations.sort(key=lambda x: x["date"], reverse=True)
    return jsonify(conversations)

@app.route("/conversations/nouvelle", methods=["POST"])
def nouvelle_conversation():
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401
    user_id = utilisateur_actuel()
    aujourdhui = datetime.now().strftime("%Y-%m-%d")
    count_aujourdhui = 0
    for fichier in os.listdir(CONV_DIR):
        if fichier.endswith(".json"):
            try:
                with open(f"{CONV_DIR}/{fichier}", "r", encoding="utf-8") as f:
                    conv = json.load(f)
                    if conv.get("user_id") == user_id and conv["date"][:10] == aujourdhui:
                        count_aujourdhui += 1
            except:
                continue
    if count_aujourdhui >= 20:
        return jsonify({"erreur": "limite atteinte"}), 403
    conv_id = str(uuid.uuid4())[:8]
    conv = {
        "id": conv_id,
        "user_id": user_id,
        "titre": "Nouvelle Novae",
        "date": datetime.now().isoformat(),
        "messages": [{"role": "system", "content": PERSONNALITE}]
    }
    sauvegarder_conversation(conv)
    return jsonify({"id": conv_id})

@app.route("/conversations/<conv_id>", methods=["GET"])
def charger_conv(conv_id):
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401
    conv = charger_conversation(conv_id)
    if not conv or conv.get("user_id") != utilisateur_actuel():
        return jsonify({"erreur": "introuvable"}), 404
    return jsonify(conv)

@app.route("/conversations/<conv_id>", methods=["DELETE"])
def supprimer_conversation(conv_id):
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401
    conv = charger_conversation(conv_id)
    if not conv or conv.get("user_id") != utilisateur_actuel():
        return jsonify({"erreur": "introuvable"}), 404
    path = f"{CONV_DIR}/{conv_id}.json"
    if os.path.exists(path):
        os.remove(path)
    return jsonify({"ok": True})

@app.route("/conversations/<conv_id>/renommer", methods=["POST"])
def renommer_conversation(conv_id):
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401
    data = request.json
    nouveau_titre = (data.get("titre") or "").strip()
    if not nouveau_titre:
        return jsonify({"erreur": "titre vide"}), 400
    conv = charger_conversation(conv_id)
    if not conv or conv.get("user_id") != utilisateur_actuel():
        return jsonify({"erreur": "introuvable"}), 404
    conv["titre"] = nouveau_titre[:60]
    sauvegarder_conversation(conv)
    return jsonify({"ok": True, "titre": conv["titre"]})

@app.route("/modele_actif", methods=["GET"])
def get_modele_actif():
    return jsonify({"model": MODEL_ACTIF})

# ============== CHANGER MODÈLE ==============

@app.route("/changer_modele", methods=["POST"])
def changer_modele():
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401
    global MODEL_ACTIF
    data = request.json
    nouveau = data.get("model", "").strip()
    if nouveau not in MODELES_DISPONIBLES:
        return jsonify({"erreur": f"Modèle inconnu : {nouveau}"}), 400
    MODEL_ACTIF = nouveau
    print(f"Modèle switché vers : {MODEL_ACTIF}")
    return jsonify({"ok": True, "model": MODEL_ACTIF})

# ============== ENVOI DE MESSAGE ==============

MAX_TOOL_ROUNDS = 3

@app.route("/envoyer", methods=["POST"])
def envoyer():
    if not utilisateur_connecte():
        return jsonify({"erreur": "non connecté"}), 401

    data = request.json
    message = data["message"]
    conv_id = data["conv_id"]
    # Liste des noms d'outils activés envoyée par le front
    # Si absente (vieux client), on active tout
    outils_actifs_front = data.get("outils_actifs", list(OUTILS_PAR_NOM.keys()))

    conv = charger_conversation(conv_id)
    if not conv or conv.get("user_id") != utilisateur_actuel():
        return jsonify({"erreur": "conversation introuvable"}), 404

    date_heure = datetime.now().strftime("%A %d %B %Y à %H:%M")
    message_avec_date = f"[Info système : nous sommes le {date_heure}]\n{message}"
    conv["messages"].append({"role": "user", "content": message_avec_date})

    reponse = ""
    thinking = ""
    rounds = 0

    while rounds < MAX_TOOL_ROUNDS:
        rounds += 1

        if MODEL_ACTIF in MODELES_OLLAMA:
            outils_session = None
            if MODEL_ACTIF not in MODELES_SANS_TOOLS:
                # Construire la liste d'outils : intersection de ce que le front
                # a activé, de ce que le modèle supporte, et des restrictions modèle
                noms_autorises = set(outils_actifs_front)
                if MODEL_ACTIF in MODELES_SANS_WEB:
                    noms_autorises -= OUTILS_INTERDITS_QWYTHOS
                outils_session = [
                    OUTILS_PAR_NOM[nom]
                    for nom in noms_autorises
                    if nom in OUTILS_PAR_NOM
                ]
                if not outils_session:
                    outils_session = None

            kwargs_chat = {
                "model": MODEL_ACTIF,
                "messages": conv["messages"],
                "options": {"temperature": 0.3},
            }
            if outils_session:
                kwargs_chat["tools"] = outils_session
            if MODEL_ACTIF in MODELES_AVEC_THINKING:
                kwargs_chat["think"] = True

            response = ollama.chat(**kwargs_chat)
        else:
            reponse = "Modèle non reconnu."
            break

        message_reponse = response["message"]
        tool_calls = message_reponse.get("tool_calls")

        thinking_round = message_reponse.get("thinking", "")
        if thinking_round and not thinking:
            thinking = thinking_round.strip()

        if not tool_calls:
            reponse = message_reponse.get("content", "")
            break

        tool_calls_serialisables = []
        for tc in tool_calls:
            tool_calls_serialisables.append({
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": dict(tc["function"]["arguments"])
                }
            })

        conv["messages"].append({
            "role": "assistant",
            "content": message_reponse.get("content", ""),
            "tool_calls": tool_calls_serialisables
        })

        for tool_call in tool_calls:
            nom_outil = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            print(f"Outil appelé : {nom_outil} avec {arguments}")
            resultat = executer_outil(nom_outil, arguments)
            conv["messages"].append({"role": "tool", "content": resultat})

    if not reponse:
        reponse = "Désolé, j'ai eu du mal à traiter ta demande. Tu peux reformuler ?"

    conv["messages"].append({"role": "assistant", "content": reponse, "thinking": thinking})

    messages_user = [m for m in conv["messages"] if m["role"] == "user"]
    if len(messages_user) == 1:
        threading.Thread(target=generer_titre_async, args=(conv_id,), daemon=True).start()

    sauvegarder_conversation(conv)

    return jsonify({
        "reponse": reponse,
        "thinking": thinking,
        "titre": conv["titre"]
    })


if __name__ == "__main__":
    app.run(debug=True)