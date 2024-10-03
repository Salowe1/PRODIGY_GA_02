from flask import Flask, render_template, request, jsonify
from io import BytesIO
import base64
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)

# Load the pre-trained model (make sure to have the correct model and pipeline)
try:
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt')

    try:
        # Generate the image
        image = model(prompt).images[0]

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({'image_data': img_str})

    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
