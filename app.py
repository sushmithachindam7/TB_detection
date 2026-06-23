# app.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, render_template_string
from PIL import Image

app = Flask(__name__)

# Load model
model = models.resnet18(weights=None)   # avoids deprecated 'pretrained'
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("tb_model.pth", map_location="cpu"))
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["Normal", "TB"]

# Homepage with colorful upload form
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
        <!doctype html>
        <html>
        <head>
            <title>TB Detection</title>
            <style>
                body {
                    background: linear-gradient(to right, #ffecd2, #fcb69f);
                    font-family: Arial, sans-serif;
                    text-align: center;
                    padding: 50px;
                }
                h1 { color: #333; margin-bottom: 20px; }
                form {
                    background: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    display: inline-block;
                }
                input[type="file"] { margin: 10px 0; }
                input[type="submit"] {
                    background: #ff6f61;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                input[type="submit"]:hover { background: #ff3b2e; }
            </style>
        </head>
        <body>
            <h1>Upload an image for prediction</h1>
            <form method="POST" action="/predict" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <br>
                <input type="submit" value="Upload">
            </form>
        </body>
        </html>
    """)

# Prediction route with error handling
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template_string("<h2 style='color:red;'>No file selected</h2>")

    file = request.files["file"]

    try:
        img = Image.open(file).convert("RGB")
    except Exception:
        return render_template_string("<h2 style='color:red;'>Invalid image file. Please upload a JPG or PNG.</h2>")

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]

    return render_template_string(f"""
        <!doctype html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    background: linear-gradient(to right, #a1c4fd, #c2e9fb);
                    font-family: Arial, sans-serif;
                    text-align: center;
                    padding: 50px;
                }}
                h1 {{
                    color: #222;
                    font-size: 28px;
                    margin-bottom: 20px;
                }}
                .result {{
                    background: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    display: inline-block;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    text-decoration: none;
                    background: #ff6f61;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                }}
                a:hover {{ background: #ff3b2e; }}
            </style>
        </head>
        <body>
            <div class="result">
                <h1>Prediction: {label}</h1>
                <a href="/">Go back</a>
            </div>
        </body>
        </html>
    """)

if __name__ == "__main__":
    app.run(debug=True)
