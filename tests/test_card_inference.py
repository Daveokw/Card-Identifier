import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import card_inference as inference


class FakeSession:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits

    def run(self, output_names, inputs):
        self.output_names = output_names
        self.inputs = inputs
        return [self.logits]


class CardInferenceTests(unittest.TestCase):
    def test_rejects_small_image(self) -> None:
        with self.assertRaisesRegex(inference.InvalidCardImage, "too small"):
            inference.prepare_image(Image.new("RGB", (64, 64), "white"))

    def test_rejects_low_contrast_image(self) -> None:
        with self.assertRaisesRegex(inference.InvalidCardImage, "not clear enough"):
            inference.prepare_image(Image.new("RGB", (256, 256), (128, 128, 128)))

    def test_prefers_pair_matching_portrait_orientation(self) -> None:
        logits = np.full((4, len(inference.LABELS)), -8.0, dtype=np.float32)
        logits[[0, 2], 3] = 8.0
        logits[[1, 3], 4] = 8.02
        fake_session = FakeSession(logits)
        pixels = np.random.default_rng(7).integers(0, 256, (320, 240, 3), dtype=np.uint8)

        with patch("card_inference.get_session", return_value=fake_session):
            prediction = inference.classify_card(Image.fromarray(pixels))

        self.assertEqual(prediction.label, "ace of spades")
        self.assertEqual(fake_session.output_names, ["logits"])
        self.assertEqual(fake_session.inputs["images"].shape, (4, 3, 224, 224))

    def test_rejects_ambiguous_model_output(self) -> None:
        logits = np.zeros((4, len(inference.LABELS)), dtype=np.float32)
        pixels = np.random.default_rng(8).integers(0, 256, (320, 240, 3), dtype=np.uint8)

        with patch("card_inference.get_session", return_value=FakeSession(logits)):
            with self.assertRaisesRegex(inference.InvalidCardImage, "No reliable result"):
                inference.classify_card(Image.fromarray(pixels))


if __name__ == "__main__":
    unittest.main()
