from flask import Flask, jsonify
from flask import request
from flask_ngrok import run_with_ngrok

import io
import torchvision.transforms as transforms
from PIL import Image

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer
import cv2
import sys

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

@app.route("/")
def home():
    return "<h1>Running</h1>"


def transform_image(image_path, model_path, label_path):
  
  class_names = [name.strip() for name in open(label_path).readlines()]
  net = create_mobilenetv1_ssd(len(class_names), is_test=True)
  net.load(model_path)

  predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
  
  orig_image = cv2.imread(image_path)
  image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
  boxes, labels, probs = predictor.predict(image, 10, 0.4) # 0.4 with classification_loss, 0.9 with dr_loss
  
  label_data = []
  for i in range(boxes.size(0)):
    label_data.append({
      'label': class_names[labels[i]],
      'class_prob': f"{probs[i]:.2f}",
    })
  return label_data;
    
  
    
@app.route('/predict')
def predict():
  image_path = request.args.get("image_path")
  model_path = request.args.get("model_path")
  label_path = request.args.get("label_path")
  label_data = transform_image(image_path, model_path, label_path)
    
  return jsonify(label_data)

app.run()