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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
    }

    .stTextArea textarea {
        font-size: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.title("🎨 AI Image Generator using Stable Diffusion")

st.markdown(
    """
Generate stunning AI images from text prompts using Stable Diffusion.
"""
)

# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    # Hugging Face Token
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password"
    )

    # Login to Hugging Face
    if hf_token:

        try:
            login(token=hf_token)
            st.success("✅ Hugging Face Login Successful")

        except Exception as e:
            st.error(f"❌ Invalid Token: {str(e)}")

    # Model Selection
    model_id = st.selectbox(
        "Select Model",
        [
            "dreamlike-art/dreamlike-diffusion-1.0",
            "runwayml/stable-diffusion-v1-5"
        ]
    )

    # Number of Images
    num_images = st.slider(
        "Number of Images",
        min_value=1,
        max_value=4,
        value=2
    )

    # Width
    width = st.slider(
        "Image Width",
        min_value=256,
        max_value=768,
        value=512,
        step=64
    )

    # Height
    height = st.slider(
        "Image Height",
        min_value=256,
        max_value=768,
        value=512,
        step=64
    )

    # Guidance Scale
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=20.0,
        value=7.5
    )

    # Inference Steps
    num_inference_steps = st.slider(
        "Inference Steps",
        min_value=10,
        max_value=100,
        value=30
    )

# ============================================================
# DEVICE SETUP
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.write(f"🖥️ Device: {device}")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_name):

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )

    pipe = pipe.to(device)

    return pipe

# ============================================================
# PROMPT SECTION
# ============================================================

st.subheader("📝 Enter Your Prompt")

default_prompt = """
A breathtaking cinematic sunrise over a peaceful tropical seashore,
golden sunlight emerging from the horizon and reflecting beautifully
across the calm ocean waves. A flock of elegant birds gracefully flying
across the glowing sun in the sky, creating a magical silhouette effect.
Crystal-clear sea water with sparkling reflections, colorful fishes
joyfully jumping above the waves near the shore. Soft morning mist,
realistic clouds painted with orange, pink, and golden tones,
gentle sea breeze atmosphere, ultra-detailed, photorealistic,
vibrant colors, dreamy lighting, highly realistic water textures,
serene and inspiring mood, 8K ultra HD, masterpiece nature photography style.
"""

prompt = st.text_area(
    "Describe Your Image",
    value=default_prompt,
    height=250
)

# ============================================================
# NEGATIVE PROMPT
# ============================================================

negative_prompt = st.text_area(
    "🚫 Negative Prompt (Optional)",
    value="blurry, low quality, distorted, ugly, watermark, text",
    height=100
)

# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button("🚀 Generate Images")

# ============================================================
# IMAGE TO BYTES
# ============================================================

def image_to_bytes(image):

    buf = BytesIO()

    image.save(buf, format="PNG")

    byte_im = buf.getvalue()

    return byte_im

# ============================================================
# GENERATE IMAGES
# ============================================================

if generate_button:

    if prompt.strip() == "":

        st.warning("⚠️ Please enter a prompt.")

    else:

        try:

            # Load Model
            with st.spinner("📦 Loading Stable Diffusion Model..."):

                pipe = load_model(model_id)

            # Generate Images
            with st.spinner("🎨 Generating AI Images..."):

                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )

                images = result.images

            st.success("✅ Images Generated Successfully!")

            st.markdown("---")

            st.subheader("🖼️ Generated Images")

            # ====================================================
            # DISPLAY IMAGES IN GRID
            # ====================================================

            cols = st.columns(2)

            for idx, image in enumerate(images):

                with cols[idx % 2]:

                    st.image(
                        image,
                        caption=f"Generated Image {idx+1}",
                        use_container_width=True
                    )

                    st.download_button(
                        label=f"⬇️ Download Image {idx+1}",
                        data=image_to_bytes(image),
                        file_name=f"generated_image_{idx+1}.png",
                        mime="image/png"
                    )

        except Exception as e:

            st.error(f"❌ Error: {str(e)}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
### 💡 Prompt Tips

✅ Use cinematic, ultra realistic, HDR, 8K keywords  
✅ Add lighting and atmosphere descriptions  
✅ Use detailed scenery descriptions  
✅ Higher inference steps improve quality  
✅ GPU strongly recommended
"""
)