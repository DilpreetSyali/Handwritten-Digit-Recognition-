import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load the model
model = tf.keras.models.load_model('handwritten_mnist_model.h5')

# Explicitly compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def preprocess_image(image_path):
    """Load and preprocess the image to match training data preprocessing."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove noise
    image = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Deskew the image
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Resize the image
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
    
    # Normalize the image
    image = image / 255.0
    
    # Reshape the image
    image = image.reshape(1, 28, 28, 1)
    
    return image
def predict_digit(image_path):
    """Predict the digit in the image and return the predicted digit."""
    image = preprocess_image(image_path)
    
    # Predict the digit
    prediction = model.predict(image)
    
    # Debugging: Print the prediction array
    print(f"Prediction array: {prediction}")
    
    predicted_digit = np.argmax(prediction)
    return predicted_digit

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_digit = predict_digit(file_path)
        messagebox.showinfo("Prediction", f"Predicted Digit: {predicted_digit}")
        display_image(file_path)

def display_image(file_path):
    image = Image.open(file_path)
    image = image.resize((200, 200), Image.LANCZOS)
    image_tk = ImageTk.PhotoImage(image)
    panel.config(image=image_tk)
    panel.image = image_tk

# Create the main application window
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("400x400")

# Style configuration
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12))
style.configure('TLabel', font=('Helvetica', 14))
style.configure('TFrame', background='#f0f0f0')

# Main frame
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Label
title_label = ttk.Label(main_frame, text="Handwritten Digit Recognition", font=("Helvetica", 18))
title_label.pack(pady=10)

# Image display panel
panel = ttk.Label(main_frame)
panel.pack(pady=20)

# Browse button
browse_button = ttk.Button(main_frame, text="Browse Image", command=browse_file)
browse_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()