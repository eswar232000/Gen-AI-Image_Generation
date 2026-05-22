import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# -----------------------------------
# Streamlit Config
# -----------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨"
)

st.title("🎨 AI Image Generator")
st.write("Generate AI images using Hugging Face Inference API")

# -----------------------------------
# Hugging Face API Token
# -----------------------------------
HF_TOKEN = "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"

# -----------------------------------
# API Configuration
# -----------------------------------
API_URL = (
    "https://api-inference.huggingface.co/models/"
    "stabilityai/stable-diffusion-2"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -----------------------------------
# Prompt Input
# -----------------------------------
prompt = st.text_area(
    "Enter your prompt",
    "A futuristic cyberpunk city at night"
)

# -----------------------------------
# Generate Image Function
# -----------------------------------
def generate_image(prompt):

    payload = {
        "inputs": prompt
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    return response.content

# -----------------------------------
# Generate Button
# -----------------------------------
if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")

    else:

        with st.spinner("Generating image..."):

            image_bytes = generate_image(prompt)

            image = Image.open(
                BytesIO(image_bytes)
            )

            st.image(
                image,
                caption="Generated Image",
                use_container_width=True
            )

            st.success(
                "Image Generated Successfully!"
            )
