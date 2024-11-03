import cv2 as cv
from ultralytics import YOLO
import streamlit as st
from streamlit_lottie import st_lottie
import json
import numpy as np

def load_lottifiel(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


st_lottie(load_lottifiel("Media/solar.json"), height=440)


# Load a model
model = YOLO('Model/v1.0/weights/best.pt')  # load an official model

def calculate_polygon_area(coordinates):
    """
    Calculate the area of a polygon using the Shoelace formula.

    Parameters:
    coordinates (list): A list of (x, y) pairs representing the vertices of the polygon.

    Returns:
    float: The area of the polygon.
    """
    area = 0.0
    n = len(coordinates)
    for i in range(n):
        j = (i + 1) % n
        area += coordinates[i][0] * coordinates[j][1]
        area -= coordinates[j][0] * coordinates[i][1]
    area = abs(area) / 2.0
    return area


def box_center(box):
    """
    Calculate the center point of a rectangle.

    The function takes a tuple or list representing a rectangle (box) defined by its top-left and bottom-right corners.
    The coordinates are expected to be in the format (x1, y1, x2, y2), where (x1, y1) are the coordinates of the top-left corner,
    and (x2, y2) are the coordinates of the bottom-right corner.

    Parameters:
    - box (tuple or list): The coordinates of the rectangle in the format (x1, y1, x2, y2).

    Returns:
    - tuple: A tuple containing the integer values of the center point (center_x, center_y) of the rectangle.
    """
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2

    return int(center_x), int(center_y)


def pixel_area_to_sqf(pixel_area, inches_per_pixel):
    """
    Convert an area from pixels to square feet.

    Parameters:
    pixel_area (float): The area in pixels.
    inches_per_pixel (float): The scale of the image in inches per pixel.

    Returns:
    float: The area in square feet.
    """
    area_sq_inches = pixel_area * (inches_per_pixel ** 2)
    area_sqf = area_sq_inches / 144  # Convert square inches to square feet
    return area_sqf


def predict_image(img, disatnce_cm, solar_size):
    if disatnce_cm is None:
        disatnce_cm = 1000
    if solar_size is None:
        solar_size = 16.5

    img = cv.resize(img, (1720, 1180))

    results = model(img, conf=0.2)
    for result in results:
        for res in result:

            box = res.boxes.xyxy[0].tolist()

            mask = np.array(res.masks.xy[0])

            pixel_area = calculate_polygon_area(mask)  # Example pixel area
            inches_per_pixel = 0.001 * disatnce_cm  # Example scale: 0.001 inches per pixel
            mask_area = round(pixel_area_to_sqf(
                pixel_area, inches_per_pixel), 2)

            img = cv.polylines(
                img, np.int32([mask]), True, (0, 0, 255), 4)

            box_center_x, box_center_y = box_center(box)

            cv.putText(img, f"{mask_area//solar_size} Solar", (box_center_x, box_center_y),
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (31, 247, 7), 3)
    return img


disatnce_cm = st.number_input(
    "", value=None, placeholder="Distace camera to object in CM..")

solar_size = st.number_input(
    "", value=None, placeholder="Per solar panel size in sqft..")

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(
        image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)

    img = predict_image(img, disatnce_cm, solar_size)

    success, encoded_image = cv.imencode('.jpg', img)
    st.image(encoded_image.tobytes(), channels="BGR", output_format="auto")
