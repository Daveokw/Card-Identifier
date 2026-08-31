"""Lightweight, shared inference utilities for the card identifier apps."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "card_model.onnx"

IMAGE_SIZE = 224
MIN_IMAGE_SIDE = 128
MIN_CONTRAST = 12.0
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 250.0
MIN_SHARPNESS = 25.0
MIN_CONFIDENCE = 0.70
MIN_MARGIN = 0.20

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

LABELS = (
    "ace of clubs", "ace of diamonds", "ace of hearts", "ace of spades",
    "eight of clubs", "eight of diamonds", "eight of hearts", "eight of spades",
    "five of clubs", "five of diamonds", "five of hearts", "five of spades",
    "four of clubs", "four of diamonds", "four of hearts", "four of spades",
    "jack of clubs", "jack of diamonds", "jack of hearts", "jack of spades",
    "joker", "king of clubs", "king of diamonds", "king of hearts",
    "king of spades", "nine of clubs", "nine of diamonds", "nine of hearts",
    "nine of spades", "queen of clubs", "queen of diamonds", "queen of hearts",
    "queen of spades", "seven of clubs", "seven of diamonds", "seven of hearts",
    "seven of spades", "six of clubs", "six of diamonds", "six of hearts",
    "six of spades", "ten of clubs", "ten of diamonds", "ten of hearts",
    "ten of spades", "three of clubs", "three of diamonds", "three of hearts",
    "three of spades", "two of clubs", "two of diamonds", "two of hearts",
    "two of spades",
)


class InvalidCardImage(ValueError):
    """Raised when an image is unsuitable for reliable classification."""


@dataclass(frozen=True)
class CardPrediction:
    label: str
    confidence: float
    margin: float


@lru_cache(maxsize=1)
def get_session() -> ort.InferenceSession:
    """Load and optimise the model once per process."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def prepare_image(image: Image.Image) -> Image.Image:
    """Correct orientation and reject visibly unusable inputs."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < MIN_IMAGE_SIDE:
        raise InvalidCardImage("The image is too small. Upload a clearer, closer photograph of one card.")

    preview = ImageOps.contain(image.convert("L"), (IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(preview, dtype=np.float32)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())

    if brightness < MIN_BRIGHTNESS:
        raise InvalidCardImage("The image is too dark. Photograph the card in better lighting.")
    if brightness > MAX_BRIGHTNESS:
        raise InvalidCardImage("The image is overexposed. Reduce glare and try again.")
    if contrast < MIN_CONTRAST:
        raise InvalidCardImage("The card is not clear enough. Use a sharper image with better contrast.")

    if pixels.shape[0] >= 3 and pixels.shape[1] >= 3:
        centre = pixels[1:-1, 1:-1]
        laplacian = (
            -4 * centre
            + pixels[:-2, 1:-1]
            + pixels[2:, 1:-1]
            + pixels[1:-1, :-2]
            + pixels[1:-1, 2:]
        )
        if float(laplacian.var()) < MIN_SHARPNESS:
            raise InvalidCardImage("The image appears blurred. Use a sharper photograph of the card.")

    return image


def _normalise(image: Image.Image) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def classify_card(image: Image.Image) -> CardPrediction:
    """Classify a card using upright and inverted orientation pairs."""
    image = prepare_image(image)
    rotations = [image.rotate(angle, expand=True) for angle in (0, 90, 180, 270)]
    batch = np.stack([_normalise(rotated) for rotated in rotations]).astype(np.float32)
    logits = get_session().run(["logits"], {"images": batch})[0]
    orientation_probabilities = _softmax(logits)

    # A card can be portrait or landscape. Average each orientation with its
    # 180-degree inverse, then retain the more decisive of the two pairs.
    paired_probabilities = np.stack(
        (
            orientation_probabilities[[0, 2]].mean(axis=0),
            orientation_probabilities[[1, 3]].mean(axis=0),
        )
    )
    expected_pair = 0 if image.height >= image.width else 1
    alternative_pair = 1 - expected_pair
    expected_strength = float(np.max(paired_probabilities[expected_pair]))
    alternative_strength = float(np.max(paired_probabilities[alternative_pair]))
    selected_pair = (
        alternative_pair
        if alternative_strength > expected_strength + 0.15
        else expected_pair
    )
    probabilities = paired_probabilities[selected_pair]

    top_indices = np.argsort(probabilities)[-2:][::-1]
    best_index, second_index = (int(index) for index in top_indices)
    confidence = float(probabilities[best_index])
    margin = confidence - float(probabilities[second_index])

    if confidence < MIN_CONFIDENCE or margin < MIN_MARGIN:
        raise InvalidCardImage(
            "No reliable result. Upload a clear photograph containing one complete playing card."
        )

    return CardPrediction(LABELS[best_index], confidence, margin)
