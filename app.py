import streamlit as st
import torch
import os

from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler
)

# ---------------------------------
# Streamlit Configuration
# ---------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 AI Image Generator")
st.write("Generate AI images using Stable Diffusion")

# ---------------------------------
# Hugging Face Token
# ---------------------------------
HF_TOKEN = "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"

# ---------------------------------
# Small CPU-Friendly Model
# ---------------------------------
MODEL_ID = "segmind/tiny-sd"

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_auth_token=HF_TOKEN
    )

    # Fix scheduler issue
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config
    )

    # Run on CPU
    pipe = pipe.to("cpu")

    # CPU optimization
    pipe.enable_attention_slicing()

    return pipe

# ---------------------------------
# Load Model
# ---------------------------------
with st.spinner("Loading AI model..."):

    pipe = load_model()

st.success("Model Loaded Successfully!")

# ---------------------------------
# User Inputs
# ---------------------------------
prompt = st.text_area(
    "Enter Prompt",
    "A futuristic cyberpunk city at night"
)

negative_prompt = st.text_input(
    "Negative Prompt",
    "blurry, low quality"
)

steps = st.slider(
    "Inference Steps",
    5,
    20,
    10
)

guidance = st.slider(
    "Guidance Scale",
    1.0,
    10.0,
    7.5
)

# ---------------------------------
# Generate Image
# ---------------------------------
if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")

    else:

        try:

            with st.spinner("Generating image..."):

                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance
                ).images[0]

                # Create folder
                os.makedirs(
                    "generated_images",
                    exist_ok=True
                )

                image_path = (
                    "generated_images/output.png"
                )

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

                st.success(
                    "Image Generated Successfully!"
                )

        except Exception as e:

            st.error(f"Error: {str(e)}")
