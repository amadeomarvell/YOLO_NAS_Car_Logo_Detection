from fastapi import FastAPI, Form
from PIL import Image
import io
import random
import torch
import cv2
import numpy as np
import base64
from starlette.responses import JSONResponse

app = FastAPI()

@app.post('/inference_folder')
def inference_from_folder(img_name: str = Form(...), conf: float = Form(...)):
    
    # Perform inference
    img_path = f"./folder_image/{img_name}"

    # img np array
    img = perform_inference(img_path, conf)

    pil_img = Image.fromarray(img)
    byte_io = io.BytesIO()
    pil_img.save(byte_io, format="JPEG")
    output_bytes = byte_io.getvalue()
    img_base64 = base64.b64encode(output_bytes).decode()
    
    data = {
        "inf_img": img_base64
    }

    return JSONResponse(content=data)


def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # Reduce font size by adjusting fontScale
        font_scale = tl / 4  # Adjust this value to further reduce the font size
        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Function to perform inference
def perform_inference(uploaded_file, confidence):
    # Load YOLO-NAS Model
    # model = models.get(
    #     'yolo_nas_m',
    #     num_classes=28,
    #     checkpoint_path='/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/ckpt_best.pth'
    # )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
     # Get class names from the model
    class_names = model.predict(np.zeros((1, 1, 3)), conf=confidence)._images_prediction_lst[0].class_names

    img_array = np.array(Image.open(uploaded_file))
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Inference logic (modify this based on your YOLO-NAS script)
    preds = model.predict(img, conf=confidence)._images_prediction_lst[0]
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)

    label_names_confidence = {}
    for box, cnf, cs in zip(bboxes, confs, labels):
        class_name = class_names[int(cs)]
        label_names_confidence[class_name] = cnf

        plot_one_box(box[:4], img, label=f'{class_name} {cnf:.3}', color=[255, 0, 0])

    return img, label_names_confidence