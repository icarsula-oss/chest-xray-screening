
from pathlib import Path
from io import BytesIO
import base64

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------
# 1. APPLICATION CONFIGURATION
# ------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "chest_xray_model.keras"

IMAGE_SIZE = (224, 224)

# ------------------------------------
# 2. LOAD TRAINED MODEL
# ------------------------------------

print("Loading trained CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

# ------------------------------------
# 3. PREPROCESS IMAGE
# ------------------------------------

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ------------------------------------
# 4. HOME PAGE
# ------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------
# 5. PREDICTION ENDPOINT
# ------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "xray_image" not in request.files:
        return render_template(
            "index.html",
            error="Please upload an image."
        ), 400

    uploaded_file = request.files["xray_image"]

    if uploaded_file.filename == "":
        return render_template(
            "index.html",
            error="No file selected."
        ), 400

    try:
        image_bytes = uploaded_file.read()

        image = Image.open(BytesIO(image_bytes))

        if image.format not in ("JPEG", "PNG"):
            raise ValueError("Unsupported image format")

        image.load()

        # Generate image preview
        preview_image = image.convert("RGB")
        preview_image.thumbnail((600, 600))

        buffer = BytesIO()
        preview_image.save(buffer, format="JPEG")

        preview = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        # Prepare input for the CNN
        processed_image = prepare_image(image)

        # Make prediction
        score = float(
            model.predict(
                processed_image,
                verbose=0
            )[0][0]
        )

        # Experimental classification threshold
        threshold = 0.5

        if score >= threshold:
            prediction = "PNEUMONIA"
            class_score = score
        else:
            prediction = "NORMAL"
            class_score = 1 - score

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=round(class_score * 100, 2),
            preview=preview
        )

    except (
        UnidentifiedImageError,
        ValueError,
        OSError,
        Image.DecompressionBombError
    ):
        return render_template(
            "index.html",
            error="Please upload a valid JPG or PNG image."
        ), 400


@app.errorhandler(413)
def too_large(error):
    return render_template(
        "index.html",
        error="Image exceeds the 8 MB upload limit."
    ), 413


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self'; "
        "form-action 'self'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    return response


# ------------------------------------
# 6. START THE WEB APPLICATION
# ------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
