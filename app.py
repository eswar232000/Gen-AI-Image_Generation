import streamlit as st
from huggingface_hub import InferenceClient
import io

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="AI Image Generator"
)

st.title("🎨 AI Image Generator")

# -----------------------------
# Read Token Securely
# -----------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

# -----------------------------
# HF Client
# -----------------------------
client = InferenceClient(
    token=HF_TOKEN
)

MODEL_NAME = "black-forest-labs/FLUX.1-schnell"

# -----------------------------
# Prompt
# -----------------------------
prompt = st.text_input(
    "Enter Prompt",
    "A futuristic cyberpunk city"
)

# -----------------------------
# Generate
# -----------------------------
if st.button("Generate Image"):

    try:

        with st.spinner("Generating image..."):

            image = client.text_to_image(
                prompt,
                model=MODEL_NAME
            )

            st.image(image)

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")

            st.download_button(
                "Download Image",
                img_bytes.getvalue(),
                "generated.png",
                "image/png"
            )

    except Exception as e:

        st.error(str(e))
