# Card Identifier

A Deep Learning project that classifies playing cards from images. Built with **PyTorch** and **Streamlit**.

## Live Demo
Try the live app here: [Card Identifier on Streamlit](https://card-identifier.streamlit.app/)

## About the Project
This project uses a **ResNet18** convolutional neural network to identify playing cards. The model has been trained on a custom dataset of playing card images, split into training, validation, and testing sets, covering **53 classes** (the standard 52-card deck + Joker).

**Note:** This application is currently a prototype. While functional, the model is not perfect and may misclassify images, particularly in poor lighting or unusual angles. Future adjustments, improvements, and contributions are highly welcomed!

**Key Features:**
*   **Deep Learning:** Utilises a pre-trained ResNet18 model fine-tuned for this specific 53-class classification task.
*   **Web Interface:** A sleek, interactive UI built with Streamlit allowing users to upload an image and get instant predictions.
*   **Desktop App:** Also includes an alternative Tkinter-based desktop GUI (`card_identifier tkinter.py`).
*   **High Accuracy:** Employs image augmentation and normalisation techniques for robust predictions.

## Technology Stack
*   **Machine Learning:** PyTorch, Torchvision
*   **Web Framework:** Streamlit
*   **Data Processing:** Pillow (PIL), Joblib
*   **Language:** Python 3.x

## How to Run Locally

1. **Clone the repository** (or download the files):
   ```bash
   git clone <your-repo-url>
   cd "Card Identifier"
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run card_identifier.py
   ```

4. *(Optional)* **Run the Tkinter Desktop App:**
   ```bash
   python "card_identifier tkinter.py"
   ```

## Dataset
The model was trained on a dataset containing thousands of images of playing cards. The dataset is organised into `train`, `valid`, and `test` directories to ensure the model generalises well to unseen images. 

## Let's Connect
Feel free to reach out if you have questions or want to collaborate!
