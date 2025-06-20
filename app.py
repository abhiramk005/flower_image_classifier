from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model and classes
model = tf.keras.models.load_model("model/flower_model.h5")
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            try:
                img = Image.open(file.stream).convert("RGB")
                input_tensor = preprocess(img)
                pred = model.predict(input_tensor)
                class_index = int(np.argmax(pred))
                confidence = float(np.max(pred)) * 100
                return jsonify({
                    "prediction": class_names[class_index],
                    "confidence": f"{confidence:.2f}%"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "No file uploaded"}), 400

    # GET request → return HTML
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
