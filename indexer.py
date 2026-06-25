"""
indexer.py - Script d'indexation de la base de connaissances Zekwo
À lancer depuis le dossier Zekwo/ chaque fois que tu ajoutes des fichiers dans knowledge/

Usage : python indexer.py
"""

import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ============== CONFIG ==============

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "static", "chroma_db")

# Modèle d'embeddings — léger, gratuit, tourne en local
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Taille des chunks (en caractères) — un chunk = un morceau de texte indexé séparément
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150  # chevauchement pour ne pas couper un raisonnement en deux

# ============== FONCTIONS ==============

def extraire_texte_pdf(path):
    """Extrait le texte brut d'un fichier PDF."""
    try:
        reader = PdfReader(path)
        texte = ""
        for page in reader.pages:
            texte += page.extract_text() or ""
        return texte.strip()
    except Exception as e:
        print(f"  ⚠️  Impossible de lire {path} : {e}")
        return ""


def extraire_texte_txt(path):
    """Lit un fichier .txt ou .md."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"  ⚠️  Impossible de lire {path} : {e}")
        return ""


def decouper_en_chunks(texte, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Découpe un texte en morceaux avec chevauchement."""
    chunks = []
    start = 0
    while start < len(texte):
        end = start + chunk_size
        chunk = texte[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def construire_metadata(filepath):
    """Construit les métadonnées d'un fichier pour retrouver sa source."""
    rel = os.path.relpath(filepath, KNOWLEDGE_DIR)
    parties = rel.replace("\\", "/").split("/")

    # Ex: maths/arithmetique/cours.pdf -> matiere=maths, chapitre=arithmetique, fichier=cours.pdf
    matiere = parties[0] if len(parties) >= 1 else "general"
    chapitre = parties[1] if len(parties) >= 2 else ""
    fichier = parties[-1]

    return {
        "source": rel.replace("\\", "/"),
        "matiere": matiere,
        "chapitre": chapitre,
        "fichier": fichier
    }


def collecter_fichiers(knowledge_dir):
    """Parcourt tous les sous-dossiers de knowledge/ et retourne les fichiers supportés."""
    fichiers = []
    extensions_ok = {".pdf", ".txt", ".md"}

    for root, dirs, files in os.walk(knowledge_dir):
        # Ignore les dossiers cachés
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions_ok:
                fichiers.append(os.path.join(root, f))

    return fichiers


def indexer():
    print("=" * 50)
    print("  Zekwo — Indexation de la base de connaissances")
    print("=" * 50)

    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"\n❌ Dossier knowledge/ introuvable : {KNOWLEDGE_DIR}")
        print("Crée le dossier et ajoutes-y des fichiers avant de lancer l'indexeur.")
        sys.exit(1)

    # Initialisation ChromaDB
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # On recrée la collection proprement à chaque indexation
    # (évite les doublons si tu relances après avoir modifié un fichier)
    try:
        client.delete_collection("knowledge")
        print("\n🔄 Collection existante supprimée (réindexation complète)")
    except:
        pass

    collection = client.create_collection(
        name="knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Collecte des fichiers
    fichiers = collecter_fichiers(KNOWLEDGE_DIR)
    if not fichiers:
        print(f"\n⚠️  Aucun fichier trouvé dans {KNOWLEDGE_DIR}")
        print("Ajoute des fichiers .txt, .md ou .pdf dans le dossier knowledge/")
        sys.exit(0)

    print(f"\n📁 {len(fichiers)} fichier(s) trouvé(s)\n")

    total_chunks = 0
    chunk_id = 0

    for filepath in fichiers:
        ext = os.path.splitext(filepath)[1].lower()
        nom_affiche = os.path.relpath(filepath, KNOWLEDGE_DIR)

        print(f"📄 {nom_affiche}")

        # Extraction du texte
        if ext == ".pdf":
            texte = extraire_texte_pdf(filepath)
        else:
            texte = extraire_texte_txt(filepath)

        if not texte:
            print("   ↳ Fichier vide ou illisible, ignoré.\n")
            continue

        # Découpage en chunks
        chunks = decouper_en_chunks(texte)
        metadata = construire_metadata(filepath)

        print(f"   ↳ {len(chunks)} chunk(s) générés (matière: {metadata['matiere']}, chapitre: {metadata['chapitre'] or 'racine'})")

        # Ajout dans ChromaDB
        ids = [f"chunk_{chunk_id + i}" for i in range(len(chunks))]
        metadatas = [metadata for _ in chunks]

        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        chunk_id += len(chunks)
        total_chunks += len(chunks)
        print()

    print("=" * 50)
    print(f"✅ Indexation terminée : {total_chunks} chunks dans la base")
    print(f"   Modèle d'embeddings : {EMBEDDING_MODEL}")
    print(f"   Base stockée dans   : {CHROMA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    indexer()
