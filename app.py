import streamlit as st
import torch
import os

from diffusers import StableDiffusionPipeline
from huggingface_hub import login

# ---------------------------------
# Hugging Face Login
# ---------------------------------
HF_TOKEN = "hf_your_token_here"

login(token=HF_TOKEN)

# ---------------------------------
# Streamlit Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 AI Image Generator")
st.write("Generate AI images from text prompts using Stable Diffusion")

# ---------------------------------
# Lightweight Stable Diffusion Model
# ---------------------------------
MODEL_ID = "OFA-Sys/small-stable-diffusion-v0"

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )

    # Run on CPU
    pipe = pipe.to("cpu")

    # CPU memory optimization
    pipe.enable_attention_slicing()

    return pipe

# ---------------------------------
# Load the Model
# ---------------------------------
with st.spinner("Loading AI model... Please wait..."):
    pipe = load_model()

st.success("Model Loaded Successfully!")

# ---------------------------------
# User Inputs
# ---------------------------------
prompt = st.text_area(
    "Enter your prompt",
    "A futuristic cyberpunk city at night, ultra realistic"
)

negative_prompt = st.text_input(
    "Negative Prompt",
    "blurry, low quality, distorted"
)

num_inference_steps = st.slider(
    "Inference Steps",
    min_value=10,
    max_value=30,
    value=20
)

guidance_scale = st.slider(
    "Guidance Scale",
    min_value=1.0,
    max_value=10.0,
    value=7.5
)

# ---------------------------------
# Generate Image
# ---------------------------------
if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a valid prompt.")

    else:

        with st.spinner("Generating image... Please wait..."):

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

            # Create output folder
            os.makedirs("generated_images", exist_ok=True)

            image_path = "generated_images/generated_image.png"

            # Save image
            image.save(image_path)

            # Display image
            st.image(
                image,
                caption="Generated Image",
                use_container_width=True
            )

            # Download button
            with open(image_path, "rb") as file:

                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )

            st.success("Image Generated Successfully!")
