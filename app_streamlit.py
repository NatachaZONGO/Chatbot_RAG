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

/* Carte de reformulation */
.reformulation-card {
    background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
    border: 1.5px solid #6366F1;
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 18px;
    margin-bottom: 6px;
}
.reformulation-card .ref-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #4F46E5;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 8px;
}
.reformulation-card .ref-text {
    font-size: 1.02rem;
    color: #1E1B4B;
    font-style: italic;
    line-height: 1.6;
}

/* Badge verdict */
.verdict-badge-pertinent {
    display: inline-block;
    background: #D1FAE5;
    color: #065F46;
    border: 1.5px solid #6EE7B7;
    border-radius: 20px;
    padding: 4px 16px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 8px;
}
.verdict-badge-ameliorer {
    display: inline-block;
    background: #FEF3C7;
    color: #92400E;
    border: 1.5px solid #FCD34D;
    border-radius: 20px;
    padding: 4px 16px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 8px;
}
.verdict-badge-nonpertinent {
    display: inline-block;
    background: #FEE2E2;
    color: #991B1B;
    border: 1.5px solid #FCA5A5;
    border-radius: 20px;
    padding: 4px 16px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 8px;
}
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
OLLAMA_MODEL = "llama3.2:1b"

# ===== OUTILS DB =====
def conn_string() -> str:
    return (
        f"dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']} "
        f"host={DB_CONFIG['host']} port={DB_CONFIG['port']}"
    )

# ===== GESTION DE LA PERSISTANCE =====
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

# ===== QUERY PARAMS =====
def set_query_params(params: dict):
    try:
        st.query_params.update(params)
    except Exception:
        st.experimental_set_query_params(**params)

def get_query_param(key: str) -> Optional[str]:
    try:
        return st.query_params.get(key)
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

# ===== FONCTIONS UTILITAIRES =====
def generer_titre_conversation(premier_message: str) -> str:
    mots = premier_message.split()[:4]
    return " ".join(mots) + ("..." if len(premier_message.split()) > 4 else "")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

def retrieve_chunks(requete: str, top_k: int = 6) -> List[Dict]:
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
        f"[Source {i+1}] Titre: {r['doc_title']} | Similarité: {r['score']:.2f}\nExtrait: {r['content'][:1200]}"
        for i, r in enumerate(retrieved)
    ])

    avg_score = sum(r["score"] for r in retrieved) / len(retrieved) if retrieved else 0
    max_score = max(r["score"] for r in retrieved) if retrieved else 0
    scores_str = " | ".join([
        f"Source {i+1} ({r['doc_title'][:25]}): {r['score']:.2f}"
        for i, r in enumerate(retrieved)
    ])

    if max_score >= 0.85:
        interpretation = (
            f"ATTENTION — La source la plus proche a un score de {max_score:.2f} (très élevé). "
            "Cela indique qu'un travail quasi-identique existe déjà dans la base. "
            "Le thème manque probablement d'originalité."
        )
    elif avg_score >= 0.50:
        interpretation = (
            f"Score moyen de {avg_score:.2f} — Des travaux proches existent dans la base. "
            "Le domaine est documenté. Vérifie si le thème apporte une contribution nouvelle "
            "par rapport aux sources citées."
        )
    else:
        interpretation = (
            f"Score moyen faible de {avg_score:.2f} — La base documentaire ne couvre pas bien ce domaine. "
            "Tu dois évaluer la qualité académique du thème de façon autonome et signaler "
            "clairement l'absence de travaux similaires dans la base."
        )

    return f"""
Tu es un assistant académique chargé d'évaluer la pertinence d'un thème de projet soumis par un étudiant.
Le thème peut appartenir à N'IMPORTE QUEL domaine académique : informatique, médecine, droit, économie, sciences sociales, agronomie, éducation, etc.

═══════════════════════════════════════════
THÈME SOUMIS PAR L'ÉTUDIANT :
{theme}
═══════════════════════════════════════════

BASE DOCUMENTAIRE — travaux déjà référencés dans le système :
{context}

Scores de similarité individuels : {scores_str}
Score moyen : {avg_score:.2f} | Score maximum : {max_score:.2f}

Interprétation des scores : {interpretation}

═══════════════════════════════════════════
TA MISSION : évaluer ce thème selon TROIS critères puis rendre un verdict clair et justifié.
═══════════════════════════════════════════

CRITÈRE 1 — QUALITÉ ACADÉMIQUE DU THÈME (juge le thème en lui-même)
• Le thème est-il clairement formulé ? L'objectif est-il compréhensible ?
• Est-il réalisable pour un étudiant (ni trop vague, ni irréaliste) ?
• A-t-il un périmètre défini (terrain, population cible, outil, période) ?
• Apporte-t-il une valeur scientifique, sociale ou pratique ?

CRITÈRE 2 — COUVERTURE PAR LA BASE DOCUMENTAIRE
• La base contient-elle des travaux dans ce domaine ? Cite les sources pertinentes.
• Si le domaine n'est PAS couvert par la base, dis-le clairement et explicitement :
  "La base documentaire ne couvre pas ce domaine. L'évaluation repose uniquement sur la qualité académique du thème."
• Ne prétends jamais que les sources sont pertinentes si elles traitent d'un autre domaine.

CRITÈRE 3 — ORIGINALITÉ PAR RAPPORT À L'EXISTANT
• Le thème ressemble-t-il à un travail déjà réalisé dans la base (score élevé) ?
• Si oui, en quoi se distingue-t-il ? Ou est-il trop identique ?
• Un thème original sur un terrain nouveau ou avec une approche nouvelle a plus de valeur.

═══════════════════════════════════════════
RÈGLES DE VERDICT (applique-les rigoureusement) :
• "Pertinent"      → Thème clair, faisable, bien délimité ET suffisamment original par rapport à l'existant.
• "À améliorer"    → Thème prometteur mais trop vague, mal délimité, ou trop proche d'un existant sans apport nouveau.
• "Non pertinent"  → Thème incompréhensible, irréalisable, ou quasi-identique à un travail déjà fait (doublon).
═══════════════════════════════════════════

FORMAT DE SORTIE OBLIGATOIRE — respecte exactement ces titres Markdown :

### Verdict
[Écris exactement l'un de ces mots : Pertinent / À améliorer / Non pertinent]
Puis explique en 3 à 4 phrases claires POURQUOI ce verdict, en te basant sur les trois critères.
Cite les sources si elles sont utiles ([Source 1], [Source 2]...).
Si les sources ne couvrent pas le domaine du thème, dis-le explicitement ici.

### Scores (/5)
- clarte_precision: x/5 — [justification courte]
- alignement_domaine: x/5 — [le thème s'inscrit-il dans un domaine académique reconnu ?]
- faisabilite: x/5 — [justification courte]
- originalite: x/5 — [par rapport aux sources existantes]
- perimetre: x/5 — [le thème est-il bien délimité ?]

### Justification
Développe ton analyse complète. Pour chaque critère (qualité académique, couverture documentaire, originalité), donne des arguments précis. Cite les sources si disponibles et pertinentes.

### Reformulation
[Propose 1 à 2 phrases CONCRÈTES qui améliorent la formulation du thème : plus précis, mieux délimité, avec un terrain ou une approche clairement définis. Cette reformulation doit être directement utilisable par l'étudiant comme nouveau thème.]

### Recommandations
- [Recommandation 1 : préciser le terrain / la population / la zone géographique]
- [Recommandation 2 : définir la méthode ou l'approche utilisée]
- [Recommandation 3 : identifier les données ou ressources nécessaires]
- [Recommandation 4 : définir des indicateurs ou résultats attendus mesurables]
- [Recommandation 5 si pertinent]

### Questions
- [Question 1 à poser à l'étudiant pour clarifier le thème]
- [Question 2]
- [Question 3]
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
        "clarte_precision":   _extract_score(answer_md, "clarte_precision") or 0,
        "alignement_domaine": _extract_score(answer_md, "alignement_domaine") or 0,
        "faisabilite":        _extract_score(answer_md, "faisabilite") or 0,
        "originalite":        _extract_score(answer_md, "originalite") or 0,
        "perimetre":          _extract_score(answer_md, "perimetre") or 0,
    }

    def extract_section(title: str) -> str:
        mm = re.search(rf"###\s*{re.escape(title)}\s*(.+?)(?:\n###|\Z)", answer_md,
                       flags=re.IGNORECASE | re.DOTALL)
        return mm.group(1).strip() if mm else ""

    justification   = extract_section("Justification")
    reformulation   = extract_section("Reformulation")
    recommandations = extract_section("Recommandations")

    # Nettoyer la reformulation : enlever les marqueurs résiduels, garder le texte pur
    if reformulation:
        # Supprimer les crochets si le LLM a laissé le template
        reformulation = re.sub(r"^\[|\]$", "", reformulation.strip())
        reformulation = reformulation.strip()

    return {
        "verdict":         verdict,
        "scores":          scores,
        "justification":   justification,
        "reformulation":   reformulation,
        "recommandations": recommandations,
    }


def afficher_carte_reformulation(reformulation: str, verdict: str):
    """
    Affiche la reformulation dans une carte visuelle mise en avant,
    avec un bouton pour réutiliser le thème amélioré comme nouvelle soumission.
    """
    if not reformulation or len(reformulation) < 10:
        return

    # Choisir l'emoji selon le verdict
    if verdict == "Pertinent":
        emoji_verdict = "✅"
        conseil = "Ton thème est pertinent. Voici une version affinée si tu souhaites le préciser davantage :"
    elif verdict == "Non pertinent":
        emoji_verdict = "❌"
        conseil = "Ton thème n'est pas pertinent dans sa forme actuelle. Voici une reformulation suggérée pour le rendre académiquement valide :"
    else:
        emoji_verdict = "🔄"
        conseil = "Ton thème mérite d'être amélioré. Voici une version reformulée et mieux délimitée :"

    st.markdown(f"""
    <div class="reformulation-card">
        <div class="ref-label">💡 Suggestion de reformulation {emoji_verdict}</div>
        <div style="font-size:0.82rem; color:#4338CA; margin-bottom:10px;">{conseil}</div>
        <div class="ref-text">« {reformulation} »</div>
    </div>
    """, unsafe_allow_html=True)

    # Bouton pour soumettre la reformulation comme nouveau thème
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("👆 Utilise ce thème amélioré pour une nouvelle analyse.")
    with col2:
        if st.button("🔁 Analyser ce thème amélioré", key=f"resubmit_{hash(reformulation)}"):
            st.session_state.resubmit_theme = reformulation
            st.rerun()


def afficher_badge_verdict(verdict: str):
    """Affiche un badge coloré selon le verdict."""
    if verdict == "Pertinent":
        css_class = "verdict-badge-pertinent"
        icon = "✅"
    elif verdict == "Non pertinent":
        css_class = "verdict-badge-nonpertinent"
        icon = "❌"
    else:
        css_class = "verdict-badge-ameliorer"
        icon = "⚠️"

    st.markdown(
        f'<span class="{css_class}">{icon} {verdict}</span>',
        unsafe_allow_html=True
    )


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
    st.markdown("### Conversations")

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
            conv_id = conv["id"]
            st.markdown(
                f"<a class='{cls}' href='?conv={conv_id}'>{title}</a>",
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
st.markdown("*Assistant intelligent de validation de thèmes académiques (Verdict • Scores • Justification • Reformulation)*")
st.markdown("---")

if not st.session_state.messages:
    st.info("👋 Décris ton thème de projet. Le système analysera sa qualité académique, vérifiera s'il a déjà été réalisé, et produira une fiche de validation complète.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("IA conversationnelle", use_container_width=True):
            st.session_state.example = "Conception d'un chatbot RAG pour valider des thèmes de projets universitaires."
            st.rerun()
        if st.button("Projet cybersécurité", use_container_width=True):
            st.session_state.example = "Détection d'attaques web et URLs malveillantes par Machine Learning avec analyse de caractéristiques."
            st.rerun()
    with col2:
        if st.button("Projet finance", use_container_width=True):
            st.session_state.example = "Prédiction du risque de défaut de crédit avec des modèles de classification et interprétabilité."
            st.rerun()
        if st.button("NLP & sentiments", use_container_width=True):
            st.session_state.example = "Analyse de sentiments sur les réseaux sociaux pour améliorer la prévision de séries temporelles (prix agricoles)."
            st.rerun()

# Afficher les messages existants
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Ré-afficher la carte de reformulation pour les messages déjà enregistrés
        if message["role"] == "assistant" and "reformulation" in message:
            afficher_carte_reformulation(message["reformulation"], message.get("verdict", "À améliorer"))

# Input chat
user_input = st.chat_input("Décris ton thème de projet (1–3 phrases, tout domaine accepté)...")

# Gérer les exemples rapides
if "example" in st.session_state:
    user_input = st.session_state.example
    del st.session_state.example

# Gérer la re-soumission depuis le bouton "Analyser ce thème amélioré"
if "resubmit_theme" in st.session_state:
    user_input = st.session_state.resubmit_theme
    del st.session_state.resubmit_theme

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyse & validation en cours..."):
            retrieved = retrieve_chunks(user_input, top_k=6)

            if not retrieved:
                st.error("Aucun chunk trouvé. Vérifie que la table chunks contient des données.")
            else:
                # Affichage des sources avec badge d'interprétation
                st.markdown("**Sources retrouvées (RAG)**")
                for i, r in enumerate(retrieved, 1):
                    score = r["score"]
                    if score >= 0.85:
                        badge = "🔴 Très proche — risque de doublon"
                    elif score >= 0.60:
                        badge = "🟡 Partiellement similaire"
                    else:
                        badge = "🟢 Domaine différent ou peu couvert"
                    with st.expander(f"{i}. {r['doc_title']} — score {score:.2f}  {badge}"):
                        st.write(r["content"][:2500])

                # Génération de la réponse
                answer = generer_validation_avec_ollama(user_input, retrieved)

                # Parse du résultat
                parsed = parse_validation(answer)

                # ── Affichage du badge verdict en haut ──
                afficher_badge_verdict(parsed["verdict"])

                # ── Affichage de la réponse complète (markdown) ──
                st.markdown(answer)

                # ── Carte de reformulation (affichage mis en avant APRÈS l'analyse) ──
                afficher_carte_reformulation(parsed["reformulation"], parsed["verdict"])

                # Sauvegarde en DB
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

                # Sauvegarde dans l'historique UI (avec reformulation et verdict pour ré-affichage)
                assistant_msg = {
                    "role": "assistant",
                    "content": answer,
                    "reformulation": parsed["reformulation"],
                    "verdict": parsed["verdict"],
                }
                st.session_state.messages.append(assistant_msg)

                if st.session_state.current_id is not None:
                    for conv in st.session_state.conversations:
                        if conv["id"] == st.session_state.current_id:
                            conv["messages"] = st.session_state.messages.copy()
                            sauvegarder_conversations(st.session_state.conversations)
                            break