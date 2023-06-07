from flask import Flask, request
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import json
import PIL.Image
from torchvision import models, transforms
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import torch
from skimage.transform import resize
import cv2


app = Flask(__name__)
CORS(app)


# Load the pre-trained model
model_preprocessing = models.segmentation.deeplabv3_resnet101(pretrained=True)
model_preprocessing.eval()

# Load the model from the .h5 file
model = load_model('flask-server\\150_160-90_keypoints_model.h5')

def resize_image(image, new_shape):
    resized_image = zoom(image, (new_shape[0] / image.shape[0], new_shape[1] / image.shape[1], 1))
    return resized_image

def preprocess_and_predict(img):
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img)

    # Resize the image while preserving the aspect ratio
    max_size = max(img_pil.size)
    aspect_ratio = img_pil.size[0] / img_pil.size[1]
    new_width = int(max_size * aspect_ratio)
    new_height = int(max_size)
    img_pil = img_pil.resize((new_width, new_height))
        
    # Define the preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        ),
    ])
        # Preprocess the image
    img_tensor = preprocess(img_pil)

        # Create a mini-batch as expected by the model
    input_batch = img_tensor.unsqueeze(0)

        # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model_preprocessing.to('cuda')

        # Get the model prediction
    with torch.no_grad():
        output = model_preprocessing(input_batch)['out'][0]
    output_predictions = output.argmax(0)

        # Resize the mask to match the original image dimensions
    resized_mask = torch.nn.functional.interpolate(output_predictions.unsqueeze(0).unsqueeze(0).float(), size=(img_pil.height, img_pil.width), mode="nearest").squeeze(0).squeeze(0)

        # Create a binary mask
    human_mask = resized_mask == 15

        # Apply the mask to the original image
    segmented_image = np.array(img_pil)
    segmented_image[~human_mask] = 0

    return segmented_image

def crop_image(image, new_shape):
    current_shape = image.shape
    y_start = (current_shape[0] - new_shape[0]) // 2
    y_end = y_start + new_shape[0]
    x_start = (current_shape[1] - new_shape[1]) // 2
    x_end = x_start + new_shape[1]
    cropped_image = image[y_start:y_end, x_start:x_end, :]
    return cropped_image

def convert_tattoo_to_rgba(image_array, alpha=255):
        # Check if the input image is already in RGBA format
        if image_array.shape[2] == 4:
            return image_array

        # Create an alpha channel with the specified value
        alpha_channel = np.full((image_array.shape[0], image_array.shape[1], 1), alpha, dtype=np.uint8)

        # Identify white pixels
        white_pixels = np.all(image_array == [1., 1., 1.], axis=-1)
        black_pixels = np.all(image_array == [0., 0., 0.], axis=-1)

        # Set alpha to 0 for white pixels
        alpha_channel[white_pixels] = 0
        alpha_channel[black_pixels] = int(0.8 * alpha)

        # Concatenate the alpha channel with the RGB image
        rgba_array = np.concatenate((image_array, alpha_channel), axis=2)

        return rgba_array

def overlay_images(background, overlay):
    # Create a copy of the background image to modify
    result = background.copy()

    # Resize the overlay image to match the background shape
    overlay_resized = overlay[:background.shape[0], :background.shape[1]]

    # Normalize the alpha values to range between 0 and 1
    overlay_alpha = overlay_resized[..., 3] / 255.0

    # Compute the weighted overlay using alpha blending
    result_alpha = 1 - (1 - overlay_alpha) * (1 - result[..., 3] / 255.0)
    result_alpha = np.clip(result_alpha, 0, 1)

    result[..., :3] = overlay_alpha[..., None] * overlay_resized[..., :3] + (1 - overlay_alpha[..., None]) * result[..., :3]
    result[..., 3] = result_alpha * 255

    return result

def invert_colors(image):
    max_value = np.max(image)
    inverted_image = max_value - image
    return inverted_image

def load_image(final_img):
    img_array = np.array(final_img)
    img = Image.fromarray(img_array)
    with BytesIO() as buffer:
        img.save(buffer, format="PNG")
        png_data = buffer.getvalue()
    # Encode the PNG image data as a base64 string
    base64_str = base64.b64encode(png_data).decode('utf-8')
    # Encode the base64 string as a JSON object
    return json.dumps({"img": base64_str})

@app.route('/upload', methods=['POST'])
def make_image():
    arm_file = request.files['arm_file']
    tattoo_file = request.files['tattoo_file']
    novaX = int(request.form['x'])
    novaY = int(request.form['y'])
    novaHeight = int(request.form['height'])
    novaWidth = int(request.form['width'])
    
    img = PIL.Image.open(arm_file).convert('RGB')
    tattoo_image = PIL.Image.open(tattoo_file).convert('RGBA')

    # Try to get the EXIF data
    try:
        exif_data = img._getexif()
        if 274 in exif_data:    # 274 is the EXIF tag for Orientation
            orientation = exif_data[274]
            # Handle the orientation
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(-90, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except:
        pass

    img_np = np.array(img)

    new_height = 1280
    new_width = img_np.shape[1] / (img_np.shape[0] / new_height)

    resized_image = resize_image(img_np, (new_height, new_width, 3))

    cropped_image = crop_image(resized_image, (1280, 720, 3))

    ## Pre-Process Arm Image
    # Load the pre-trained model
    model_preprocessing = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model_preprocessing.eval()

    preprocessed_image = preprocess_and_predict(np.array(cropped_image)) #pass it to function

    # Resize the image using the resize function
    resized_pre_image = resize(preprocessed_image, (160, 90, 3), mode='constant')

    # Preprocessing Completed
    ## Predict keypoint
    from IPython.display import Image

    # add an extra dimension for the batch
    image_batch = np.expand_dims(resized_pre_image, axis=0)

    # Make predictions
    predictions = model.predict(image_batch)
    # Reshape the predicted keypoints array to a (4, 2) matrix
    predicted_keypoints = np.array(predictions).reshape((4, 2))

    # Read Tattoo Image
    from PIL import Image

    scaled_tattoo = tattoo_image.resize((int(novaWidth), int(novaHeight)))

    # Create a white background image to represent the arm
    background_color = (255, 255, 255)  # White color in RGB
    background_image = Image.new("RGB", (180, 320), background_color)

    # Overlay the image onto the canvas using PIL
    background_image.paste(scaled_tattoo, (novaX, novaY))

    # Convert the final image to a NumPy array
    image_array = np.array(background_image)

    # Ensure image is within [0, 1] range
    image_array = image_array / 255.0

    # changing it to cv2
    resized_image = resize(image_array, (320, 180, 3), mode='constant')
    ## Warp Tattoo Image

    # Define the threshold value
    threshold = 0.5

    # Binarize the image array
    #binarized_image_array = np.where(resized_image > threshold, 0., 1.)
    binarized_image_array = invert_colors(resized_image)

    # making the keypoints for the flat arm
    tl = (71, 137)
    bl = (63, 320)
    tr = (108, 137)
    br = (121, 320)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[predicted_keypoints[0][0], predicted_keypoints[0][1]], [predicted_keypoints[3][0], predicted_keypoints[3][1]], [predicted_keypoints[1][0], predicted_keypoints[1][1]], [predicted_keypoints[2][0], predicted_keypoints[2][1]]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(binarized_image_array, matrix, (720, 1280))

    # Define the threshold value
    threshold = 0.5

    # Create the mask
    mask = np.zeros((1280, 720), dtype=np.uint8)

    # Define the keypoints as a polygon
    # tl, tr, bl, br
    keypoints = np.float32([[predicted_keypoints[0][0], predicted_keypoints[0][1]],
                            [predicted_keypoints[1][0], predicted_keypoints[1][1]],
                            [predicted_keypoints[2][0], predicted_keypoints[2][1]],
                            [predicted_keypoints[3][0], predicted_keypoints[3][1]]])

    cv2.fillPoly(mask, [keypoints.astype(np.int32)], 255)

    # Apply the mask to the transformed frame
    masked_transformed_frame = cv2.bitwise_and(transformed_frame, transformed_frame, mask=mask)

    # Binarize the image array
    new_one = np.where(masked_transformed_frame > threshold, 0., 1.) ##invert colours

    image_with_alpha = convert_tattoo_to_rgba(new_one)

    rgba_arm = convert_tattoo_to_rgba(cropped_image)

    overlaid_image = overlay_images(rgba_arm, image_with_alpha)

    return load_image(overlaid_image)

if __name__ == "__main__":
    app.run(debug=True)