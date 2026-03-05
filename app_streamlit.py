# app_streamlit.py
import streamlit as st
from sentence_transformers import SentenceTransformer
import psycopg
import numpy as np
import ollama
from typing import List, Dict, Optional
from datetime import datetime
import json
import os
import uuid
import re

# Configuration de la page
st.set_page_config(
    page_title="CheiTacha Validator – RAG System AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #F7F7F8; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #ECECF1 !important;
    border-right: 1px solid #D9D9E3;
}
[data-testid="stSidebar"] * { color: #111827 !important; }

/* Boutons sidebar */
[data-testid="stSidebar"] .stButton button {
    background-color: white !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    padding: 0.6rem !important;
    transition: 0.2s !important;
}
[data-testid="stSidebar"] .stButton button:hover { background-color: #F3F4F6 !important; }

/* LISTE conversations (pas boutons) */
.conv-item {
    padding: 10px 10px;
    border-radius: 10px;
    margin: 4px 0;
    display: block;
    text-decoration: none;
    color: #111827;
    font-size: 14px;
    line-height: 1.2;
}
.conv-item:hover { background: #F3F4F6; }
.conv-active { background: #E5E7EB; font-weight: 600; }

/* CHAT */
[data-testid="stChatMessage"] { padding: 1rem 0 !important; }
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #E5E7EB !important;
    border-radius: 14px !important;
    padding: 0.9rem 1.2rem !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: white !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

/* Input */
.stChatInputContainer {
    background-color: white !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 16px !important;
}

/* Titres */
h1, h2, h3 { color: #111827 !important; }
</style>
""", unsafe_allow_html=True)

# ===== CONFIGURATION =====
DB_CONFIG = {
    "dbname": "chatbot_rag",
    "user": "postgres",
    "password": "t17@ACHA",
    "host": "localhost",
    "port": "5432"
}

CONVERSATIONS_FILE = "conversations_history.json"
OLLAMA_MODEL = "llama3.2:1b"  # si tu as mieux (3b/8b), remplace ici

# ===== OUTILS DB =====
def conn_string() -> str:
    return (
        f"dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']} "
        f"host={DB_CONFIG['host']} port={DB_CONFIG['port']}"
    )

# ===== GESTION DE LA PERSISTANCE (conversations UI) =====
def charger_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def sauvegarder_conversations(conversations):
    try:
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erreur de sauvegarde : {e}")

# ===== QUERY PARAMS (liste cliquable style ChatGPT) =====
def set_query_params(params: dict):
    """Compat Streamlit: nouveaux/anciens APIs"""
    try:
        st.query_params.update(params)  # Streamlit récent
    except Exception:
        st.experimental_set_query_params(**params)

def get_query_param(key: str) -> Optional[str]:
    try:
        return st.query_params.get(key)  # Streamlit récent
    except Exception:
        qp = st.experimental_get_query_params()
        v = qp.get(key, [None])
        return v[0] if v else None

# ===== CHARGEMENT DU MODÈLE =====
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with st.spinner("Chargement..."):
    embedding_model = load_model()

# ===== FONCTIONS (RAG + VALIDATION) =====
def generer_titre_conversation(premier_message: str) -> str:
    mots = premier_message.split()[:4]
    return " ".join(mots) + ("..." if len(premier_message.split()) > 4 else "")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

def retrieve_chunks(requete: str, top_k: int = 6) -> List[Dict]:
    """Récupère les meilleurs chunks (RAG) depuis PostgreSQL et calcule la similarité en Python."""
    q_emb = np.array(embedding_model.encode(requete), dtype=np.float32)

    try:
        with psycopg.connect(conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.id, c.content, c.embedding, d.title
                    FROM chunks c
                    JOIN documents d ON d.id = c.doc_id
                """)
                rows = cur.fetchall()
    except Exception as e:
        st.error(f"Erreur DB retrieve_chunks : {e}")
        return []

    scored: List[Dict] = []
    for chunk_id, content, emb_list, doc_title in rows:
        emb = np.array(emb_list, dtype=np.float32)
        score = cosine_similarity(q_emb, emb)
        scored.append({
            "chunk_id": int(chunk_id),
            "doc_title": str(doc_title),
            "content": str(content),
            "score": float(score),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def build_validation_prompt(theme: str, retrieved: List[Dict]) -> str:
    context = "\n\n".join([
        f"[Source {i+1}] Document: {r['doc_title']} | Similarité: {r['score']:.2f}\nExtrait: {r['content'][:1200]}"
        for i, r in enumerate(retrieved)
    ])

    return f"""
Tu es un assistant universitaire qui analyse et valide la pertinence d'un thème de projet (Machine Learning / Data Mining).
Tu dois être strict, clair, et justifier tes conclusions uniquement à partir des SOURCES fournies.

THEME A ANALYSER:
{theme}

SOURCES (extraits RAG):
{context}

Tâche:
1) Donne un VERDICT: "Pertinent" OU "À améliorer" OU "Non pertinent"
2) Donne des SCORES sur 5 pour:
   - clarte_precision
   - alignement_ml_dm
   - faisabilite
   - originalite
   - perimetre
3) Justifie chaque score (et le verdict) en citant [Source 1], [Source 2], etc.
4) Propose une REFORMULATION améliorée (1 phrase).
5) Propose 3 à 6 RECOMMANDATIONS concrètes (objectifs, données possibles, méthode ML, métriques d'évaluation).
6) Si le thème est trop vague, ajoute une section "Questions" avec 3 à 5 questions à poser à l'étudiant.

Format de sortie OBLIGATOIRE (Markdown):
### Verdict
...
### Scores (/5)
- clarte_precision: x/5
- alignement_ml_dm: x/5
- faisabilite: x/5
- originalite: x/5
- perimetre: x/5
### Justification
...
### Reformulation
...
### Recommandations
- ...
### Questions
- ...
""".strip()

def generer_validation_avec_ollama(theme: str, retrieved: List[Dict]) -> str:
    prompt = build_validation_prompt(theme, retrieved)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

def _extract_score(text: str, key: str) -> Optional[int]:
    pattern = rf"{re.escape(key)}\s*:\s*(\d)\s*/\s*5"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        v = int(m.group(1))
        return max(0, min(5, v))
    except:
        return None

def parse_validation(answer_md: str) -> Dict:
    verdict = "À améliorer"
    m = re.search(r"###\s*Verdict\s*(.+?)(?:\n###|\Z)", answer_md, flags=re.IGNORECASE | re.DOTALL)
    if m:
        v = m.group(1).strip().splitlines()[0].strip()
        v_low = v.lower()
        if "non" in v_low and "pertinent" in v_low:
            verdict = "Non pertinent"
        elif "pertinent" in v_low:
            verdict = "Pertinent"
        elif "améli" in v_low or "amel" in v_low:
            verdict = "À améliorer"
        else:
            verdict = v[:40]

    scores = {
        "clarte_precision": _extract_score(answer_md, "clarte_precision") or 0,
        "alignement_ml_dm": _extract_score(answer_md, "alignement_ml_dm") or 0,
        "faisabilite": _extract_score(answer_md, "faisabilite") or 0,
        "originalite": _extract_score(answer_md, "originalite") or 0,
        "perimetre": _extract_score(answer_md, "perimetre") or 0,
    }

    def extract_section(title: str) -> str:
        mm = re.search(rf"###\s*{re.escape(title)}\s*(.+?)(?:\n###|\Z)", answer_md,
                       flags=re.IGNORECASE | re.DOTALL)
        return mm.group(1).strip() if mm else ""

    justification = extract_section("Justification")
    reformulation = extract_section("Reformulation")
    recommandations = extract_section("Recommandations")

    return {
        "verdict": verdict,
        "scores": scores,
        "justification": justification,
        "reformulation": reformulation,
        "recommandations": recommandations,
    }

def save_analysis(
    session_id: str,
    theme_text: str,
    verdict: str,
    scores: Dict,
    justification: str,
    reformulation: str,
    recommandations: str,
    sources: List[Dict],
):
    with psycopg.connect(conn_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analyses (session_id, theme_text, verdict, scores, justification, reformulation, recommandations)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                RETURNING id
                """,
                (
                    session_id,
                    theme_text,
                    verdict,
                    json.dumps(scores, ensure_ascii=False),
                    justification or "",
                    reformulation or "",
                    recommandations or "",
                ),
            )
            analysis_id = cur.fetchone()[0]

            for rank, s in enumerate(sources, start=1):
                cur.execute(
                    """
                    INSERT INTO analysis_sources (analysis_id, chunk_id, similarity, rank)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (analysis_id, s["chunk_id"], float(s["score"]), rank),
                )
        conn.commit()

# ===== INITIALISATION =====
if "conversations" not in st.session_state:
    st.session_state.conversations = charger_conversations()

if "current_id" not in st.session_state:
    st.session_state.current_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🧠 CheiTacha AI")

    # Nouvelle conversation
    if st.button("✚ Nouvelle conversation", use_container_width=True):
        if st.session_state.messages:
            premier_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Conversation")
            titre = generer_titre_conversation(premier_msg)

            existe = False
            if st.session_state.current_id is not None:
                for conv in st.session_state.conversations:
                    if conv["id"] == st.session_state.current_id:
                        conv["messages"] = st.session_state.messages.copy()
                        conv["date"] = datetime.now().strftime("%d/%m/%Y %H:%M")
                        existe = True
                        break

            if not existe:
                new_id = max([c["id"] for c in st.session_state.conversations], default=-1) + 1
                st.session_state.conversations.insert(0, {
                    "id": new_id,
                    "titre": titre,
                    "messages": st.session_state.messages.copy(),
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M")
                })

            sauvegarder_conversations(st.session_state.conversations)

        st.session_state.messages = []
        st.session_state.current_id = None
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")

    # ===== LISTE CONVERSATIONS (style ChatGPT) =====
    st.markdown("### Conversations")

    # clic via URL (?conv=ID)
    clicked_id = get_query_param("conv")
    if clicked_id is not None:
        try:
            clicked_id_int = int(clicked_id)
            for conv in st.session_state.conversations:
                if conv["id"] == clicked_id_int:
                    st.session_state.messages = conv["messages"].copy()
                    st.session_state.current_id = conv["id"]
                    break
            set_query_params({})
            st.rerun()
        except:
            pass

    if st.session_state.conversations:
        for conv in st.session_state.conversations[:30]:
            is_active = conv["id"] == st.session_state.current_id
            cls = "conv-item conv-active" if is_active else "conv-item"

            title = conv["titre"]
            if len(title) > 28:
                title = title[:28] + "..."

            st.markdown(
                f"<a class='{cls}' href='?conv={conv['id']}'>{title}</a>",
                unsafe_allow_html=True
            )
    else:
        st.caption("Aucune conversation pour le moment.")

    st.markdown("---")
    if st.button("🗑️ Effacer tout l'historique", use_container_width=True):
        st.session_state.conversations = []
        st.session_state.messages = []
        st.session_state.current_id = None
        st.session_state.session_id = str(uuid.uuid4())
        sauvegarder_conversations([])
        st.rerun()

    st.caption(f"{len(st.session_state.conversations)} conversation(s)")

# ===== INTERFACE PRINCIPALE =====
st.markdown("""
<div style="text-align:center; margin-bottom:20px;">
    <div style="font-size:42px;">🧠</div>
</div>
""", unsafe_allow_html=True)

st.markdown("## 🧠 CheiTacha Validator – RAG System AI")
st.markdown("*Assistant intelligent de validation de thèmes (Verdict • Scores • Justification • Reformulation)*")
st.markdown("---")

# Message d'accueil + exemples
if not st.session_state.messages:
    st.info("👋 Décris ton thème de projet. Le système va retrouver des sources (RAG) et produire une fiche de validation.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("IA conversationnelle", use_container_width=True):
            st.session_state.example = "Conception d’un chatbot RAG pour valider des thèmes de projets universitaires."
            st.rerun()
        if st.button("Projet cybersécurité", use_container_width=True):
            st.session_state.example = "Détection d’attaques web et URLs malveillantes par Machine Learning avec analyse de caractéristiques."
            st.rerun()
    with col2:
        if st.button("Projet finance", use_container_width=True):
            st.session_state.example = "Prédiction du risque de défaut de crédit avec des modèles de classification et interprétabilité."
            st.rerun()
        if st.button("NLP & sentiments", use_container_width=True):
            st.session_state.example = "Analyse de sentiments sur les réseaux sociaux pour améliorer la prévision de séries temporelles (prix agricoles)."
            st.rerun()

# Afficher messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input chat
user_input = st.chat_input("Décris ton thème (1–3 phrases)...")

if "example" in st.session_state:
    user_input = st.session_state.example
    del st.session_state.example

if user_input:
    # Message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Réponse assistant (RAG + validation)
    with st.chat_message("assistant"):
        with st.spinner("Analyse & validation (RAG)..."):
            retrieved = retrieve_chunks(user_input, top_k=6)

            if not retrieved:
                st.error("Aucun chunk trouvé. Vérifie que la table chunks contient des données.")
            else:
                st.markdown("**Sources retrouvées (RAG)**")
                for i, r in enumerate(retrieved, 1):
                    with st.expander(f"{i}. {r['doc_title']} — score {r['score']:.2f}"):
                        st.write(r["content"][:2500])

                answer = generer_validation_avec_ollama(user_input, retrieved)
                st.markdown(answer)

                # Parse + save in DB
                parsed = parse_validation(answer)
                try:
                    save_analysis(
                        session_id=st.session_state.session_id,
                        theme_text=user_input,
                        verdict=parsed["verdict"],
                        scores=parsed["scores"],
                        justification=parsed["justification"],
                        reformulation=parsed["reformulation"],
                        recommandations=parsed["recommandations"],
                        sources=retrieved,
                    )
                except Exception as e:
                    st.warning(f"⚠️ Analyse non sauvegardée (DB) : {e}")

                # UI history
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Sauvegarder conversation UI
                if st.session_state.current_id is not None:
                    for conv in st.session_state.conversations:
                        if conv["id"] == st.session_state.current_id:
                            conv["messages"] = st.session_state.messages.copy()
                            sauvegarder_conversations(st.session_state.conversations)
                            break