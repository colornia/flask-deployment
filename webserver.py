import io
from io import  BytesIO
import os
import requests
from detection import models
from torchvision import transforms
from PIL import Image
from flask import Flask, request,render_template,send_file

app = Flask(__name__,template_folder='templates')  # 固定写法
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG'}
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
themodel = models.U2Net()
themodel.eval()
def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.40760392, 0.4595686, 0.48501961],
                std=[0.225, 0.224, 0.229]
    ),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = themodel(tensor)
    return outputs
@app.route("/")
def index():
    return render_template('up.html')
@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method == 'POST':  # 接收传输的图片
        img_bytes = request.files['photo'].read()
    # Preprocess the image and infrence it.
    final = get_prediction(image_bytes=img_bytes)
    img = final[0].squeeze(0)
    transs = transforms.ToPILImage()
    img = transs(img)
    byte_io =BytesIO()
    img.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')

if __name__ == "__main__":
        app.run()
