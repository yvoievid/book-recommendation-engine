import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

EXAMPLES = [
    "What It Takes: Lessons in the Pursuit of Excellence by Stephen A. Schwarzman",
    "The Practice: Shipping Creative Work by Seth Godin",
    "The Ideal Team Player: How to Recognize and Cultivate the Three Essential Virtues by Patrick Lencioni",
    "4 Essential Keys to Effective Communication in Love, Life, Work — Anywhere! by Bento C. Leal III",
    "The Undocumented Americans by Karla Cornejo Villavicencio",
    "Emotional Agility: Get Unstuck, Embrace Change, and Thrive in Work and Life by Susan David, Ph.D.",
    "The Origin of Species by Charles Darwin",
    "Stay Positive: Encouraging Quotes and Messages to Fuel Your Life with Positive Energy by Jon Gordon, Daniel Decker",
    "Chasing Failure: How Falling Short Sets You Up for Success by Ryan Leak",
    "Platform Revolution: How Networked Markets Are Transforming the Economy and How to Make Them Work for You by Geoffrey G. Parker, Marshall W. Van Alstyne, Sangeet Paul Choudary",
    "My Own Words by Ruth Bader Ginsburg, Mary Hartnett, and Wendy W. Williams",
    "Better Than Before: Mastering the Habits of Our Everyday Lives by Gretchen Rubin"
]


@st.cache_data
def load_categories():
    df = pd.read_csv("data/categories.csv", header=None)
    return df[0].tolist()

@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer("yuriivoievidka/microsoft_mpnet-base-librarian")
    cats = load_categories()
    emb = model.encode(cats, convert_to_numpy=True, show_progress_bar=False)
    # normalize once
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return model, cats, emb


if 'prompt' not in st.session_state:
    st.session_state['prompt'] = ""
if 'likes' not in st.session_state:
    st.session_state['likes'] = []
if 'dislikes' not in st.session_state:
    st.session_state['dislikes'] = []
if 'base_emb' not in st.session_state:
    st.session_state['base_emb'] = None
if 'round' not in st.session_state:
    st.session_state['round'] = 0

model, all_cats, cat_emb = load_model_and_embeddings()


st.title("📚 Book Category Recommender with Feedback")

st.markdown("**Pick an example book:**")
cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, key=f"ex_{i}"):
        st.session_state['prompt'] = example

prompt = st.text_input(
    "Enter book title or description:",
    value=st.session_state['prompt']
)
st.session_state['prompt'] = prompt

if prompt:
    q_emb = model.encode([prompt], convert_to_numpy=True, show_progress_bar=False)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    st.session_state['base_emb'] = q_emb
else:
    st.session_state['base_emb'] = None


def get_topk(embedding, k=15):
    sims = (embedding @ cat_emb.T).flatten()
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def show_recommendations(emb, round_idx):
    topk_idx, _ = get_topk(emb)
    st.subheader(f"Top 15 Categories (round {round_idx}):")
    for idx in topk_idx:
        cat = all_cats[idx]
        c1, c2, c3 = st.columns([4, 1, 1])
        c1.write(cat)
        like_key    = f"like_{round_idx}_{idx}"
        dislike_key = f"dislike_{round_idx}_{idx}"
        if c2.button("👍", key=like_key):
            if cat not in st.session_state['likes']:
                st.session_state['likes'].append(cat)
        if c3.button("👎", key=dislike_key):
            if cat not in st.session_state['dislikes']:
                st.session_state['dislikes'].append(cat)

if st.session_state['base_emb'] is not None:
    show_recommendations(st.session_state['base_emb'], st.session_state['round'])

def show_feedback():
    if st.session_state['likes']:
        st.markdown("**You liked:**")
        st.write(st.session_state['likes'])
    if st.session_state['dislikes']:
        st.markdown("**You disliked:**")
        st.write(st.session_state['dislikes'])

show_feedback()

if st.session_state['likes'] or st.session_state['dislikes']:
    if st.button("🔄 Refine Recommendations"):
        st.session_state['round'] += 1

        emb = st.session_state['base_emb'].copy()
        pos_w, neg_w = 0.7, 0.5

        if st.session_state['likes']:
            like_embs = model.encode(
                st.session_state['likes'], convert_to_numpy=True, show_progress_bar=False
            )
            like_embs = like_embs / np.linalg.norm(like_embs, axis=1, keepdims=True)
            mean_like = like_embs.mean(axis=0, keepdims=True)
            mean_like = mean_like / np.linalg.norm(mean_like)
            emb = emb + pos_w * mean_like

        if st.session_state['dislikes']:
            dis_embs = model.encode(
                st.session_state['dislikes'], convert_to_numpy=True, show_progress_bar=False
            )
            dis_embs = dis_embs / np.linalg.norm(dis_embs, axis=1, keepdims=True)
            mean_dis = dis_embs.mean(axis=0, keepdims=True)
            mean_dis = mean_dis / np.linalg.norm(mean_dis)
            emb = emb - neg_w * mean_dis

        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        show_recommendations(emb, st.session_state['round'])
        show_feedback()

st.markdown("---")
st.write("Built with Streamlit and Sentence-Transformers 🔥")
