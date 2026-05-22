import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨"
)

st.title("🎨 AI Image Generator")

# -----------------------------
# Hugging Face Token
# -----------------------------
HF_TOKEN = "hf_your_token_here"

# -----------------------------
# Tiny Model
# -----------------------------
MODEL_ID = "segmind/tiny-sd"

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        use_auth_token=HF_TOKEN
    )

    pipe = pipe.to("cpu")

    return pipe

with st.spinner("Loading model..."):
    pipe = load_model()

st.success("Model Loaded!")

# -----------------------------
# Prompt Input
# -----------------------------
prompt = st.text_input(
    "Enter Prompt",
    "A futuristic city at sunset"
)

# -----------------------------
# Generate Image
# -----------------------------
if st.button("Generate"):

    with st.spinner("Generating image..."):

        image = pipe(prompt).images[0]

        st.image(
            image,
            caption="Generated Image",
            use_container_width=True
        )
