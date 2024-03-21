import tkinter as tk
from tkinter import font as tkFont
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# GUI settings
WINDOW_WIDTH = 800 
WINDOW_HEIGHT = 600  
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400
BG_COLOR = "#000000"
BUTTON_COLOR = "#555555"
BUTTON_FONT_COLOR = "#FFFFFF"
CANVAS_COLOR = "#FFFFFF"
FONT_NAME = "Helvetica"
FONT_SIZE = 12
TITLE_FONT_SIZE = 24

root = tk.Tk()
root.title('Digit Recognizer')
root.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')  # Set the window size

# Use custom font
customFont = tkFont.Font(family=FONT_NAME, size=FONT_SIZE)
titleFont = tkFont.Font(family=FONT_NAME, size=TITLE_FONT_SIZE)

# Function to switch to drawing pane
def show_drawing_pane():
    home_frame.pack_forget()
    main_frame.pack(fill=tk.BOTH, expand=True)

# Home page frame
home_frame = tk.Frame(root, bg=BG_COLOR)
home_frame.pack(fill=tk.BOTH, expand=True)

title_label = tk.Label(home_frame, text="Digit Recognizer", font=titleFont, bg=BG_COLOR, fg=BUTTON_FONT_COLOR)
title_label.pack(pady=(100, 20))

start_button = tk.Button(home_frame, text="Start", command=show_drawing_pane, bg=BUTTON_COLOR, fg=BUTTON_FONT_COLOR, font=customFont)
start_button.pack(pady=(0, 150))  # Adjusted padding for balance

# Adding developer name label at the bottom of the home frame
developer_label = tk.Label(home_frame, text="Developed by Heshani Samadhi", font=(FONT_NAME, FONT_SIZE), bg=BG_COLOR, fg=BUTTON_FONT_COLOR)
developer_label.pack(side=tk.BOTTOM, pady=10)

# Main drawing frame (initially hidden)
main_frame = tk.Frame(root, bg=BG_COLOR)

canvas = tk.Canvas(main_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg=CANVAS_COLOR, bd=0, highlightthickness=0)
canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

button_frame = tk.Frame(main_frame, bg=BG_COLOR)
button_frame.pack(side=tk.TOP, pady=10, padx=10, fill=tk.X)

prediction_text = tk.StringVar()

def predict_digit():
    # Convert the image in canvas to a numpy array
    filename = "canvas_image.png"
    canvas.postscript(file=filename, colormode='color')
    img = Image.open(filename)
    img = img.convert('RGB').convert('L')
    img = img.resize((28, 28))

    # Invert the image (black background, white digit)
    img = ImageOps.invert(img)

    # Normalizing of the image
    img_array = np.array(img)
    img_array = img_array / 255.0

    # Reshape the image for the model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    # Update the prediction text
    prediction_text.set('Prediction: ' + str(digit))

def clear_canvas():
    canvas.delete('all')
    prediction_text.set('')

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)

# Binding paint function to the canvas
canvas.bind('<B1-Motion>', paint)

predict_button = tk.Button(button_frame, text='Predict', command=predict_digit, bg=BUTTON_COLOR, fg=BUTTON_FONT_COLOR, font=customFont)
predict_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

clear_button = tk.Button(button_frame, text='Clear', command=clear_canvas, bg=BUTTON_COLOR, fg=BUTTON_FONT_COLOR, font=customFont)
clear_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

prediction_label = tk.Label(button_frame, textvariable=prediction_text, bg=BG_COLOR, fg=BUTTON_FONT_COLOR, font=customFont)
prediction_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

# Start the app on the home page
root.mainloop()
