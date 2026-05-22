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

    .block-container {
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.title("🎨 AI Image Generator")

st.markdown(
    """
Generate AI images from text prompts using Stable Diffusion.
"""
)

# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    # ========================================================
    # HUGGING FACE TOKEN
    # ========================================================

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

            st.error(f"❌ Invalid Token: {str(e)}")

    # ========================================================
    # MODEL
    # ========================================================

    model_id = st.selectbox(
        "Select Model",
        [
            "runwayml/stable-diffusion-v1-5"
        ]
    )

    # ========================================================
    # IMAGE SETTINGS
    # ========================================================

    num_images = st.slider(
        "Number of Images",
        min_value=1,
        max_value=1,
        value=1
    )

    width = st.slider(
        "Image Width",
        min_value=256,
        max_value=384,
        value=384,
        step=64
    )

    height = st.slider(
        "Image Height",
        min_value=256,
        max_value=384,
        value=384,
        step=64
    )

    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=15.0,
        value=7.5
    )

    num_inference_steps = st.slider(
        "Inference Steps",
        min_value=10,
        max_value=30,
        value=20
    )

# ============================================================
# DEVICE SETUP
# ============================================================

device = "cpu"

st.sidebar.write(f"🖥️ Device: {device}")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_name):

    try:

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True
        )

        pipe = pipe.to(device)

        # Memory optimization
        pipe.enable_attention_slicing()

        return pipe

    except Exception as e:

        st.error(f"❌ Error Loading Model: {str(e)}")

        return None

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
serene and inspiring mood, masterpiece nature photography style.
"""

prompt = st.text_area(
    "Describe Your Image",
    value=default_prompt,
    height=220
)

# ============================================================
# NEGATIVE PROMPT
# ============================================================

negative_prompt = st.text_area(
    "🚫 Negative Prompt",
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

            # ====================================================
            # LOAD MODEL
            # ====================================================

            with st.spinner("📦 Loading Model..."):

                pipe = load_model(model_id)

            if pipe is None:

                st.stop()

            # ====================================================
            # GENERATE IMAGE
            # ====================================================

            with st.spinner("🎨 Generating Image..."):

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

            st.success("✅ Image Generated Successfully!")

            st.markdown("---")

            st.subheader("🖼️ Generated Image")

            # ====================================================
            # DISPLAY IMAGE
            # ====================================================

            for idx, image in enumerate(images):

                st.image(
                    image,
                    caption=f"Generated Image {idx + 1}",
                    use_container_width=True
                )

                st.download_button(
                    label=f"⬇️ Download Image {idx + 1}",
                    data=image_to_bytes(image),
                    file_name=f"generated_image_{idx+1}.png",
                    mime="image/png"
                )

        except torch.cuda.OutOfMemoryError:

            st.error(
                "❌ Memory Error. Reduce image size or inference steps."
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

✅ Use cinematic, HDR, ultra realistic keywords  
✅ Add lighting and atmosphere descriptions  
✅ Use detailed scenery descriptions  
✅ Smaller image sizes run faster on CPU  
"""
)
