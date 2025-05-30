import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask import redirect, url_for

app = Flask(__name__)
model = load_model('fingerprint2_spoof_model.h5')
IMG_SIZE = (96, 103)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = preprocess_image(file_path)
        prediction = model.predict(img)[0][0]
        label = "Altered (Fake)" if prediction < 0.5 else "Real"

        return render_template('result.html', filename=filename, label=label)

    return render_template('home.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
