import cv2
import numpy as np
from scipy.ndimage import sobel, filters
import matplotlib.pyplot as plt

def harris_corners(img, window_size=3, k=0.04):
    # Get the height and width of the input image
    H, W = img.shape
    # Create a window of ones for local computations
    window = np.ones((window_size, window_size))
    # Initialize the matrix to store Harris corner responses
    response = np.zeros((H, W))
    # Compute x and y derivatives using Sobel operators

    dx = sobel(img, axis=1, mode='constant')# Derivative along x-axis
    dy = sobel(img, axis=0, mode='constant')# Derivative along y-axis
    # Compute products of derivatives at each pixel
    Ix2 = dx * dx
    Iy2 = dy * dy
    Ixy = dx * dy
    # Convolve products of derivatives with the window
    Ix2_sum = filters.convolve(Ix2, window, mode='constant', cval=0)
    Iy2_sum = filters.convolve(Iy2, window, mode='constant', cval=0)
    Ixy_sum = filters.convolve(Ixy, window, mode='constant', cval=0)
    # Calculate the Harris response for each pixel
    for y in range(H):
        for x in range(W):
     # Construct the matrix M using sums of squared derivatives
            M = np.array([[Ix2_sum[y, x], Ixy_sum[y, x]],
                          [Ixy_sum[y, x], Iy2_sum[y, x]]])
        # Calculate determinant and trace of M
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
        # Compute Harris response using the formula
            response[y, x] = det_M - k * (trace_M ** 2)

    return response

# Load the image
image_path = 'Coral_colony_photo_1_year_ago.jpg'  # Replace with the actual path to your image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply the Harris Corners algorithm
response = harris_corners(img)

# Visualize the Harris response map
plt.imshow(response, cmap='gray')
plt.title("Harris Corner Response")
plt.show()
