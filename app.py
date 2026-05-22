import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 AI Image Generator")
st.write("Generate images using Stable Diffusion")

# ---------------------------------
# Hugging Face Token
# ---------------------------------
HF_TOKEN = "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"

# ---------------------------------
# Create HF Client
# ---------------------------------
client = InferenceClient(
    token=HF_TOKEN
)

# ---------------------------------
# User Prompt
# ---------------------------------
prompt = st.text_area(
    "Enter your prompt",
    "A futuristic cyberpunk city at night"
)

# ---------------------------------
# Generate Button
# ---------------------------------
if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")

    else:

        try:

            with st.spinner("Generating image..."):

                image = client.text_to_image(
                    prompt,
                    model="stabilityai/stable-diffusion-2"
                )

                # Display image
                st.image(
                    image,
                    caption="Generated Image",
                    use_container_width=True
                )

                # Save image to bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")

                # Download button
                st.download_button(
                    label="Download Image",
                    data=img_bytes.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png"
                )

                st.success(
                    "Image Generated Successfully!"
                )

        except Exception as e:

            st.error(f"Error: {str(e)}")
