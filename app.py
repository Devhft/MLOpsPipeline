from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import joblib

app = Flask(__name__)

model = joblib.load("savedmodel.pth")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]

        image = Image.open(file).convert("L")
        image = image.resize((64, 64))

        image_array = np.array(image).flatten()
        image_array = image_array.reshape(1, -1)

        prediction = model.predict(image_array)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)