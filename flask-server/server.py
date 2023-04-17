from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import json

app = Flask(__name__)
CORS(app)

def load_image(arm_img, star_img):
    arm_img = resize_img(arm_img, 10)
    star_img = cv2.resize(star_img, (arm_img.shape[1], arm_img.shape[0]))
    # Convert arm_img to RGBA
    b, g, r = cv2.split(arm_img)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    arm_img_rgba = cv2.merge((b, g, r, alpha))
    #final_img = cv2.addWeighted(arm_img_rgba, 1, star_img, 1, 0)
    # Extract the alpha channel from the tattoo image (star_img)
    alpha_channel = star_img[:, :, 3] / 255.0
    # Overlay the tattoo image onto the arm image using the extracted alpha channel
    final_img = overlay_image(arm_img_rgba, star_img, alpha_channel)
    img_array = np.array(final_img)
    img = Image.fromarray(img_array)
    with BytesIO() as buffer:
        img.save(buffer, format="PNG")
        png_data = buffer.getvalue()
    # Encode the PNG image data as a base64 string
    base64_str = base64.b64encode(png_data).decode('utf-8')
    # Encode the base64 string as a JSON object
    return json.dumps({"img": base64_str})



def overlay_image(background, foreground, alpha_channel):
    # Create an empty result image with the same size and data type as the background
    result = np.zeros_like(background)
    # Compute the inverse of the alpha channel
    inverse_alpha = 1.0 - alpha_channel
    # Loop over the color channels (ignoring the alpha channel)
    for c in range(0, 3):
        # Compute the blended color values for each channel
        result[:, :, c] = (alpha_channel * foreground[:, :, c] +
                           inverse_alpha * background[:, :, c])
    # Merge the blended color channels with the original alpha channel from the background
    result = cv2.merge((result[:, :, 0], result[:, :, 1], result[:, :, 2], background[:, :, 3]))
    return result



def resize_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)



@app.route('/upload', methods=['POST'])
def make_image():
    arm_file = request.files['arm_file']
    tattoo_file = request.files['tattoo_file']

    def read_file(file):
        img_data = file.read()
        # convert the image data to a numpy array using np.frombuffer()
        img_array = np.frombuffer(img_data, np.uint8)
        # decode the numpy array into a cv2 image using cv2.imdecode()
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # Convert BGR image to RGB
        return img
    
    # Convert white background to transparent
    tattoo_img = read_file(tattoo_file)
    # Convert the image to RGBA format
    image_rgba = cv2.cvtColor(tattoo_img, cv2.COLOR_BGR2RGBA)
    # Set the alpha channel of white pixels to 0 (transparent)
    white_pixels = np.all(image_rgba == [255, 255, 255, 255], axis=-1)
    image_rgba[white_pixels, -1] = 0
    rgb_arm = cv2.cvtColor(read_file(arm_file), cv2.COLOR_BGR2RGB)
    return load_image(rgb_arm, image_rgba)

if __name__ == "__main__":
    app.run(debug=True)


