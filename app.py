import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Stable Diffusion Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 AI Image Generator using Stable Diffusion")
st.write("Generate images from text prompts using Stable Diffusion on CPU")

# -----------------------------
# Model Loading
# -----------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )

    # Force CPU
    pipe = pipe.to("cpu")

    return pipe

with st.spinner("Loading Stable Diffusion model... Please wait..."):
    pipe = load_model()

st.success("Model Loaded Successfully!")

# -----------------------------
# User Inputs
# -----------------------------
prompt = st.text_area(
    "Enter your prompt",
    "A futuristic cyberpunk city at night, ultra realistic"
)

negative_prompt = st.text_input(
    "Negative Prompt (Optional)",
    "blurry, low quality, distorted"
)

num_inference_steps = st.slider(
    "Inference Steps",
    min_value=10,
    max_value=50,
    value=25
)

guidance_scale = st.slider(
    "Guidance Scale",
    min_value=1.0,
    max_value=15.0,
    value=7.5
)

generate_btn = st.button("Generate Image")

# -----------------------------
# Image Generation
# -----------------------------
if generate_btn:

    if prompt.strip() == "":
        st.warning("Please enter a valid prompt.")
    else:

        with st.spinner("Generating image..."):

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

            # Create output directory
            os.makedirs("generated_images", exist_ok=True)

            image_path = "generated_images/generated_image.png"

            # Save image
            image.save(image_path)

            st.image(image, caption="Generated Image", use_container_width=True)

            # Download Button
            with open(image_path, "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )

            st.success("Image Generated Successfully!")
