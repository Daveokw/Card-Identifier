"""Optional desktop interface for Card Identifier."""

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError

from card_inference import InvalidCardImage, classify_card


def upload_and_classify() -> None:
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.gif *.tif *.tiff")]
    )
    if not file_path:
        return

    try:
        with Image.open(file_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        display_image = ImageOps.contain(image, (320, 320))
        photo = ImageTk.PhotoImage(display_image)
        image_panel.configure(image=photo)
        image_panel.image = photo

        prediction = classify_card(image)
        result_label.configure(text=f"Predicted card: {prediction.label.title()}", fg="#137333")
    except InvalidCardImage as error:
        result_label.configure(text=str(error), fg="#b3261e")
    except (UnidentifiedImageError, OSError):
        messagebox.showerror("Invalid image", "The selected file could not be read as an image.")
    except Exception:
        messagebox.showerror("Analysis failed", "The image could not be analysed. Try another image.")


root = tk.Tk()
root.title("Card Identifier")
root.geometry("600x520")
root.configure(bg="#f5f5f5")

tk.Label(
    root,
    text="Card Identifier",
    font=("Arial", 18, "bold"),
    bg="#f5f5f5",
    fg="#333333",
).pack(pady=20)

image_panel = tk.Label(root, bg="#f5f5f5")
image_panel.pack()

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 13),
    wraplength=520,
    bg="#f5f5f5",
    fg="#174ea6",
)
result_label.pack(pady=20)

tk.Button(
    root,
    text="Upload card image",
    command=upload_and_classify,
    font=("Arial", 12),
    bg="#0078d7",
    fg="white",
).pack(pady=10)

if __name__ == "__main__":
    root.mainloop()
