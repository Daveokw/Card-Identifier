# Card Identifier

A lightweight computer vision application that identifies a playing card from an uploaded photograph. It covers the standard 52-card deck and Joker.

## Live demo

[Open Card Identifier on Streamlit](https://card-identifier.streamlit.app/)

## How it works

The application uses a 53-class ResNet18 model exported to ONNX for efficient CPU inference. Before classification, it corrects image orientation and checks basic image size, lighting, contrast, and sharpness. It evaluates portrait and landscape orientation pairs to make predictions less sensitive to camera rotation.

Unclear or ambiguous inputs are declined instead of being assigned a forced label. The interface displays only the predicted card, without exposing raw model scores.

## Features

- Identifies 52 standard playing cards and Joker
- Handles common image orientations automatically
- Rejects images that are too small, dark, bright, blurred, or ambiguous
- Processes uploaded images in memory
- Uses cached, CPU-based ONNX Runtime inference for Streamlit Community Cloud
- Includes an optional Tkinter desktop interface

## Run locally

```bash
git clone https://github.com/Daveokw/Card-Identifier.git
cd Card-Identifier
pip install -r requirements.txt
streamlit run card_identifier.py
```

To run the optional desktop interface:

```bash
python "card_identifier tkinter.py"
```

## Limitations

The model may still make mistakes, particularly when a card is partly hidden, highly stylised, or visually different from its training data. Use a clear photograph containing one complete card. Model performance should be measured on a representative held-out dataset before the application is used in a consequential setting.
