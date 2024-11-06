import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to perform the 2D DFT
def compute_dft(image):
    # Apply 2D Fast Fourier Transform (FFT)
    dft = np.fft.fft2(image)

    # Shift the zero-frequency component to the center
    dft_shifted = np.fft.fftshift(dft)

    # Calculate the magnitude spectrum (log scale for better visualization)
    magnitude = np.abs(dft_shifted)

    # Apply log to enhance visibility (log scale)
    magnitude_log = np.log(magnitude + 1)  # Adding 1 to avoid log(0)

    return magnitude_log

# Function to read image and split into RGB channels
def load_image(image_path):
    img = Image.open(image_path)

    # Convert image to numpy array
    img_array = np.array(img)

    return img_array

# Path to your image
image_path = "./Images/IcyLandscape.png"  # Change this to the image path you want to test

# Load and process the image
image = load_image(image_path)

# Split the image into RGB channels
r_channel, g_channel, b_channel = image[:,:,0], image[:,:,1], image[:,:,2]

# Compute the DFT of each channel
dft_r = compute_dft(r_channel)
dft_g = compute_dft(g_channel)
dft_b = compute_dft(b_channel)

# Plot the original image and the DFTs for each channel
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# DFT Magnitude for Red channel
axes[0, 1].imshow(dft_r, cmap='gray')
axes[0, 1].set_title("Red Channel DFT")
axes[0, 1].axis('off')

# DFT Magnitude for Green channel
axes[0, 2].imshow(dft_g, cmap='gray')
axes[0, 2].set_title("Green Channel DFT")
axes[0, 2].axis('off')

# DFT Magnitude for Blue channel
axes[1, 0].imshow(dft_b, cmap='gray')
axes[1, 0].set_title("Blue Channel DFT")
axes[1, 0].axis('off')

# Combine the DFT results for visualization
combined_dft = np.stack([dft_r, dft_g, dft_b], axis=-1)
axes[1, 1].imshow(np.log(np.mean(combined_dft, axis=-1) + 1), cmap='gray')  # Combining DFTs for an overall view
axes[1, 1].set_title("Average DFT Magnitude")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
