import streamlit as st
from huggingface_hub import InferenceClient
import io

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
# HF Client
# ---------------------------------
client = InferenceClient(
    token=HF_TOKEN
)

# ---------------------------------
# Public Working Model
# ---------------------------------
MODEL_NAME = "black-forest-labs/FLUX.1-schnell"

# ---------------------------------
# Prompt Input
# ---------------------------------
prompt = st.text_area(
    "Enter Prompt",
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
                    model=MODEL_NAME
                )

                # Display image
                st.image(
                    image,
                    caption="Generated Image",
                    use_container_width=True
                )

                # Download
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")

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
