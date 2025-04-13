import streamlit as st
import torch
import clip
import pickle
import json
import numpy as np
from PIL import Image
import openai

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load embeddings + metadata
with open("lfw_clip_embeddings.pkl", "rb") as f:
    embeddings, metadata = pickle.load(f)
embeddings = np.array(embeddings)

# OpenAI setup
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🔍 Tinder Scanner - Face Match & Risk Estimator")
st.markdown("""
Upload a photo of your partner. We'll compare it with a database of real faces.
If they look suspiciously like someone on a dating app... you’ll find out.
""")

# Upload form
uploaded_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess & embed
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = model.encode_image(img_tensor).cpu().numpy()

    # Cosine similarity
    similarities = np.dot(embeddings, img_embedding.T).flatten()
    top_indices = similarities.argsort()[::-1][:5]
    top_matches = [(similarities[i], metadata[i]) for i in top_indices]

    st.subheader("🧬 Top Visual Matches")
    for score, match in top_matches:
        st.image(match["image_path"], caption=f"{match['name']} (Score: {score:.2f})", width=150)

    # Simple cheating risk score
    risk_score = int(min(100, (np.mean([s for s, _ in top_matches]) - 0.20) * 250))
    st.markdown(f"### 🚨 Cheating Risk Score: **{risk_score}/100**")

    # OpenAI Explanation
    with st.spinner("🔍 AI analyzing matches..."):
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a suspicious but clever AI that explains visual face matches in a dating-app scenario."},
                {"role": "user", "content": f"""I uploaded a photo and the top visual matches are:
{[m['name'] for _, m in top_matches]}.
What can you tell me? Is this person potentially active on Tinder? Be analytical, insightful, and a little cheeky.""",
                }
            ]
        )
        explanation = response.choices[0].message.content
        st.markdown("### 🧠 AI Analysis")
        st.write(explanation)
