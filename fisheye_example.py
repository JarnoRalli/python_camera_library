import numpy as np
import omnidirectional_camera as ocam
import cv2
import matplotlib.pyplot as plt

# This is the "focal length" of the camera, you can test with different values
focal_length = 200

# These are the coefficients that describe the fisheye camera (based on the omnidirectional model)
model_coefficients = np.array(
    [
        730.949123,
        315.876984,
        -177.960849,
        -352.468231,
        -678.144608,
        -615.917273,
        -262.086205,
        -42.961956,
    ]
)

# Load the test fish-eye camera image
image = cv2.imread(".//test_data/fish_eye_camera/test_fisheye.jpg")

# Calculate the look-up-table for converting the image into perspective camera image
x, y = ocam.perspective_lut(
    image.shape, (505.480427, 381.777786), focal_length, model_coefficients
)

# Convert the image into perspective camera image
image_perspective = cv2.remap(image, x, y, cv2.INTER_LINEAR)

# Show the re-mapped image
plt.imshow(image_perspective)
plt.show()
