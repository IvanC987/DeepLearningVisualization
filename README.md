# DeepLearningVisualization

This is a simple project that's designed to visualize image approximation using a neural network with **Polynomial Expansion (PE)** terms. 
<br>
The purpose of this is for me to familiarize myself with working on images to prepare for my next project: Image Diffusion
<br>
This project includes image generation through snapshots during training, which is saved to a specified directory, and can later be compiled into a video that gives a visualization of how the images are being modeled.



## Files
- [PolyExp.py](#polyexppy)- Defines and trains a neural network to approximate a given image
  - Key Features:
    - Initializes model weights and learning rate.
    - Iteratively generates images based on polynomial terms.
    - Saves generated images at specified evaluation interval.
- [ImageTransform.py](#imagetransformpy)- Prepares the input image by resizing, cropping, and/or padding to better fit model
  - Key Features:
    - Prompts user to adjust provided image if the image is not square
    - Pads image based on `patch_size`
- [CompileVideo.py](#compilevideopy)- Compiles the saved images from `PolyExp.py` into a video (currently using mp4)
  - Key Features:
    - Allows adjustable FPS and frame-stepping options to balance video length and convergence detail
    - Sorts images by time for chronologically ordered video generation



## Notes and Limitations
- Square Images: Non-square images may introduce artifacts. Squaring, resizing and/or cropping is recommended.
- Convergence Pattern: The generated images often converge from the bottom-right to the top-left due to coordinate initialization


<br>

### Mandelbrot Images:
- Mandelbrot Images: Sourced from mandelbrot.site, used for testing and demonstration purposes.
https://mandelbrot.site/#:~:text=Explore%20the%20infinite%20complexity%20of%20the%20Mandelbrot%20Set