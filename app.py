import streamlit as st
import torch

from diffusers import StableDiffusionPipeline

# ---------------------------------
# Streamlit Config
# ---------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨"
)

st.title("🎨 AI Image Generator")

# ---------------------------------
# Hugging Face Token
# ---------------------------------
HF_TOKEN = "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"

# ---------------------------------
# Tiny Stable Diffusion Model
# ---------------------------------
MODEL_ID = "segmind/tiny-sd"

# ---------------------------------
# Load Model
# ---------------------------------
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

# ---------------------------------
# Prompt
# ---------------------------------
prompt = st.text_input(
    "Enter Prompt",
    "A futuristic cyberpunk city"
)

# ---------------------------------
# Generate
# ---------------------------------
if st.button("Generate Image"):

    with st.spinner("Generating..."):

        image = pipe(prompt).images[0]

        st.image(
            image,
            caption="Generated Image",
            use_container_width=True
        )
