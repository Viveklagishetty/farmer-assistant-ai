from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

# Define the classes dictionary
classes = {
    0: "Alluvial Soil:-{ Rice, Wheat, Sugarcane, Maize, Cotton, Soybean, Jute }",
    1: "Black Soil:-{ Virginia, Wheat, Jowar, Millets, Linseed, Castor, Sunflower }",
    2: "Clay Soil:-{ Rice, Lettuce, Chard, Broccoli, Cabbage, Snap Beans }",
    3: "Red Soil:{ Cotton, Wheat, Pulses, Millets, Oilseeds, Potatoes }"
}

def model_predict(image_path, model):
        image = load_img(image_path, target_size=(150, 150))
        image = img_to_array(image)
        image = image / 255
        image = np.expand_dims(image, axis=0)
        result = np.argmax(model.predict(image))
        prediction = classes[result]
        
        output_page = None
        if result == 0:
            output_page = "Alluvial.html"
        elif result == 1:
            output_page = "Black.html"
        elif result == 2:
            output_page = "Clay.html"
        elif result == 3:
            output_page = "Red.html"
        
        return prediction, output_page

 






