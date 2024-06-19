from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model(r'C:\Users\Windows 10 Pro\Downloads\plant_disease_classifier.keras')
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize the image to match the model's input shape
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    if predicted_class == 0:
        return "Healthy"
    elif predicted_class == 1:
        return "Powdery"
    else:
        return "Rust"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            prediction = process_image(file_path)
    
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
