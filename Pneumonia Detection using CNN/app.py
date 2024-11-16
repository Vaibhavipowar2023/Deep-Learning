from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image



# Initialize the Flask application
app = Flask(__name__)

# Load the VGG19 model and customize it
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
x = base_model.output
flat = Flatten()(x)
drop_out = Dropout(0.5)(flat)  # Adding dropout layer for regularization
class_1 = Dense(4608, activation='relu')(drop_out)
class_2 = Dense(1152, activation='relu')(class_1)
output = Dense(2, activation='softmax')(class_2)

# Create a new model based on the modified VGG19 architecture
model_03 = Model(inputs=base_model.inputs, outputs=output)

# Load the pre-trained weights (ensure this path is correct)
model_03.load_weights('model.keras')  # Load the best model

print('Model loaded. Check http://127.0.0.1:5000/')

# Function to map class numbers to class names
def get_class_name(class_no):
    """Map class number to corresponding class name."""
    if class_no == 0:
        return "Normal"
    elif class_no == 1:
        return "Pneumonia"
    else:
        return "Unknown"

# Function to process and predict the image
def get_result(img_path):
    """Process and predict the image class."""
    img = Image.open(img_path)  # Open the image using PIL
    img = img.resize((128, 128))  # Resize the image to 128x128
    img = np.array(img)  # Convert image to numpy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image to [0, 1] range

    result = model_03.predict(img)  # Predict the class of the image
    result_class = np.argmax(result, axis=1)  # Get the class with the highest probability
    return result_class

# Route to render the index page (home page)
@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the index page with upload form and prediction result."""
    prediction_result = None  # Default result is None
    if request.method == 'POST':
        f = request.files['file']  # Get the uploaded file
        # Define the path to save the uploaded file
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        
        # Create 'uploads' directory if it does not exist
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Save the file securely
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        
        # Get prediction result
        class_no = get_result(file_path)
        prediction_result = get_class_name(class_no[0])  # Map the result class number to class name
    
    return render_template('index.html', prediction_result=prediction_result)

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
