import gradio as gr
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2
from blurgenerator import lens_blur_with_depth_map

# torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_depth_image(image):
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


def blur_with_depth_image(image, depth_image, components, exposure_gamma):
    cv_image = convert_from_image_to_cv2(image)
    cv_depth_image = convert_from_image_to_cv2(depth_image)
    neg_cv_depth_image = ~cv_depth_image

    # apply blur
    cv_img = lens_blur_with_depth_map(
        img=cv_image,
        depth_map=neg_cv_depth_image,
        components=components,
        exposure_gamma=exposure_gamma,
        num_layers=10,
        min_blur=1,
        max_blur=100
    )
    img = convert_from_cv2_to_image(cv_img)

    return img


def process_image(image, components, exposure_gamma, state):
    if state is None or not state['image'] == image:
        state = dict(
            image=image,
            cv_image=None,
            cv_depth_map=None
        )

    if state['cv_image'] is None:
        state['cv_image'] = convert_from_image_to_cv2(state['image'])

    if state['cv_depth_map'] is None:
        depth_map = get_depth_image(state['image'])
        state['cv_depth_map'] = convert_from_image_to_cv2(depth_map)

    neg_cv_depth_map = ~state['cv_depth_map']

    # apply blur
    cv_blur_img = lens_blur_with_depth_map(
        img=state['cv_image'],
        depth_map=neg_cv_depth_map,
        components=components,
        exposure_gamma=exposure_gamma,
        num_layers=10,
        min_blur=1,
        max_blur=100
    )
    img = convert_from_cv2_to_image(cv_blur_img)

    return img, state


with gr.Blocks() as app:
    state_image = gr.State(None)
    with gr.Row():
        with gr.Column():
            input_image = gr.inputs.Image(type="pil", label="source")
            btn_get_depth = gr.Button("Estimate depth")
            depth_image = gr.inputs.Image(type="pil", label="predicted depth")
            # paint = gr.Paint()
            # paint = gr.ImageMask()
            # paint = gr.ImagePaint()
        with gr.Column():
            components = gr.Slider(minimum=1, maximum=10,
                                   step=1, label="components")
            exposure_gamma = gr.Slider(
                minimum=1, maximum=10, step=1, label="exposure gamma")
            btn_process = gr.Button("Focus subject")
            output_image = gr.outputs.Image(
                type="pil", label="focused")

    btn_get_depth.click(
        get_depth_image,
        [input_image],
        [depth_image]
    )

    btn_process.click(
        blur_with_depth_image,
        [input_image, depth_image, components, exposure_gamma],
        [output_image]
    )
app.launch()
