import gradio as gr
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image

# torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def process_image(image):
    # prepare image for the model
    encoding = feature_extractor(image, return_tensors="pt")
    
    # forward pass
    with torch.no_grad():
       outputs = model(**encoding)
       predicted_depth = outputs.predicted_depth
    
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                 ).squeeze()
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype('uint8')
    img = Image.fromarray(formatted)
    return img
    
    return result
    
title = "Subject focus"
description = "Create focus to subject using depth estimation with DPT."
examples =[[]]

iface = gr.Interface(fn=process_image, 
                     inputs=gr.inputs.Image(type="pil"), 
                     outputs=gr.outputs.Image(type="pil", label="predicted depth"),
                     title=title,
                     description=description,
                     examples=examples,
                     enable_queue=True)
iface.launch(debug=True)