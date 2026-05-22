import streamlit as st
import torch

from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from io import BytesIO

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🎨 AI Image Generator")

st.write("Generate AI images from text prompts.")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    hf_token = st.text_input(
        "Hugging Face Token",
        type="password"
    )

    if hf_token and "hf_logged" not in st.session_state:

        try:

            login(token=hf_token)

            st.session_state["hf_logged"] = True

            st.success("✅ Login Successful")

        except Exception as e:

            st.error(str(e))

    guidance_scale = st.slider(
        "Guidance Scale",
        1.0,
        15.0,
        7.5
    )

    num_inference_steps = st.slider(
        "Inference Steps",
        10,
        30,
        20
    )

# ============================================================
# CPU DEVICE
# ============================================================

device = "cpu"

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )

    pipe = pipe.to(device)

    pipe.enable_attention_slicing()

    return pipe

# ============================================================
# PROMPT
# ============================================================

prompt = st.text_area(
    "Enter Prompt",
    height=180
)

# ============================================================
# BUTTON
# ============================================================

generate = st.button("🚀 Generate")

# ============================================================
# IMAGE TO BYTES
# ============================================================

def image_to_bytes(image):

    buf = BytesIO()

    image.save(buf, format="PNG")

    return buf.getvalue()

# ============================================================
# GENERATE IMAGE
# ============================================================

if generate:

    if prompt.strip() == "":

        st.warning("Enter prompt")

    else:

        try:

            with st.spinner("Loading model..."):

                pipe = load_model()

            with st.spinner("Generating image..."):

                image = pipe(
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=384,
                    width=384
                ).images[0]

            st.image(
                image,
                caption="Generated Image",
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download",
                data=image_to_bytes(image),
                file_name="generated.png",
                mime="image/png"
            )

        except Exception as e:

            st.error(f"Error: {str(e)}")
