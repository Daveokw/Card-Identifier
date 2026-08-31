"""Streamlit interface for Card Identifier."""

import logging

import streamlit as st
from PIL import Image, UnidentifiedImageError

from card_inference import InvalidCardImage, classify_card, get_session

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000

st.set_page_config(page_title="Card Identifier", page_icon="🂡", layout="centered")
st.title("🂡 Card Identifier")
st.write("Upload a clear photograph containing one playing card.")

st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a card image",
    type=["jpg", "jpeg", "jpe", "jfif", "png", "webp", "bmp", "dib", "gif", "tif", "tiff"],
)

if uploaded_file is not None:
    try:
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            raise InvalidCardImage("The image is too large. Upload a file smaller than 20 MB.")

        image = Image.open(uploaded_file)
        image.load()
        st.image(image, caption="Uploaded image", width="stretch")

        with st.spinner("Identifying card…"):
            get_session()
            prediction = classify_card(image)

        st.success(f"Predicted card: {prediction.label.title()}")
    except InvalidCardImage as error:
        st.warning(str(error))
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Upload a valid image and try again.")
    except Exception:
        LOGGER.exception("Unexpected card-classification failure")
        st.error("The image could not be analysed. Please try another image.")
