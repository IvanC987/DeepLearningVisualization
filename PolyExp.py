import torch
from torch import nn
from torch.optim import AdamW
from ImageTransform import get_image_data, create_image
import time
import os



"""
Notes: 

1. In the docs/comments, the Polynomial Expansion would be referred to as PE
2. As of right now, using non-square images, i.e. 4:3 aspect ratios and such, would introduce artifacts in generated images
    Currently suspecting the reason to be patch misalignment, but not yet certain. Will leave it be for now until future revision
    If provided image is not square, Compression and Cropping based on the smaller dimension will be offered
3. The generated images tend to converge from bottom-right to top-left. This is likely due to the initialization of the
    pixel_coords variable where (x, y) values are lowest and highest in the top-left and bottom-right respectively. 
    This effect should be mitigated in darker images in the outer areas. 
4. The effect noted above can be removed by using random values for pixel_coords, but due to certain reasons this will be left as such

"""


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}")


# Hyperparameters
# -----------------------------------
num_terms = 5  # Number of polynomial terms to use
training_iterations = 450
eval_interval = 10  # Set to 1 at final testing to create snapshot at every iteration

# `start_noisy`- Starts out with randomized RGB values by randomizing bias parameter in range (0, 255)
# Although it does not converge well, it can be fun to play around with
start_noisy = False
patch_size = 20
image_path = "./Images/Star.jpg"
save_image_dir = "./CreatedImages/PE_Images/"
os.makedirs(save_image_dir, exist_ok=True)
# -----------------------------------


del_files = input(f"Clear {save_image_dir} Folder? [Y/N]: ")
while del_files.lower() not in ["y", "n"]:
    print("Invalid input.")
    del_files = input(f"Clear {save_image_dir} Folder? [Y/N]: ")

if del_files.lower() == "y":
    for item in os.listdir(save_image_dir):
        item_path = os.path.join(save_image_dir, item)
        os.remove(item_path)

    print(f"\nFiles deleted in {save_image_dir=}\n")
print()


def init_data() -> tuple[tuple, torch.tensor, torch.tensor]:
    """
    Initialize the target image's data, dimensions, and starting pixel coordinates

    :return: image dimensions, initialized pixel coordinates, and target data in forms of torch tensors
    """
    # This returns the image as a PyTorch tensor of shape (num_pixels, color) and its corresponding image_dimensions after adjustment
    image_data, image_dimensions = get_image_data(image_path, patch_size)

    # pixel_coords.shape=(num_pixels, 2)
    pixel_coords = torch.tensor([[(x, y) for y in range(image_dimensions[1])] for x in range(image_dimensions[0])], dtype=torch.float32).reshape(-1, 2)

    # This one below makes the 'coordinates' random, which removes the bottom-right to top-left convergence, making the adjustments more 'uniform', but converges slower
    # pixel_coords = torch.rand((image_dimensions[0] * image_dimensions[1], 2), dtype=torch.float)

    # Normalize the coordinate values in the range of (0, 1), otherwise exponentiation would cause overflow
    pixel_coords[:, 0] /= image_dimensions[0]
    pixel_coords[:, 1] /= image_dimensions[1]

    # Shift the range to (1, 2) for faster training
    # Otherwise exponentiation of very low values close to 0 would tilt towards the bias more than coefficients during training
    # Note that the naming of this variable is a bit of a misnomer, but I'll leave it as is
    pixel_coords += 1

    # Since image_data is of shape width * height, where each element is (R, G, B) value of each pixel, this would need to be reshaped into the desired format
    # Which would be splitting the image into the shape (color, width, height)
    red_data = image_data[:, 0]
    green_data = image_data[:, 1]
    blue_data = image_data[:, 2]

    # Stacking the tensors and normalizing the values to be between 0 and 1
    target_data = torch.stack((red_data, green_data, blue_data))

    return image_dimensions, pixel_coords, target_data


def update_lr(iteration: int, base_lr=0.1, min_lr=0.01, warmup_iterations=100, decay_factor=4e4) -> None:
    """
    Updates the learning rate of the optimizer based on current iteration/step

    :param iteration: Current iteration/step in training loop
    :param base_lr: Base learning rate
    :param min_lr: Minimum learning rate
    :param warmup_iterations: Number of iterations to use base_lr before decaying
    :param decay_factor: The factor that controls how fast learning rate decays
    :return: None
    """

    # I'm just using a simple negative sqrt curve as the decaying function. Adjust as needed.
    # With default value of base_lr=0.1 and decay_factor=4e4, it will decay for 324 iterations before reaching min_lr
    # Can be visualized using Desmos     y=-sqrt(x/decay_factor) + (base_lr - min_lr)
    if iteration < warmup_iterations:
        lr = base_lr  # Use the base learning rate during the warmup period
    else:
        # Calculate the decreasing learning rate after warmup
        decay = -((iteration - warmup_iterations)/decay_factor)**0.5 + base_lr
        lr = max(min_lr, decay)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # print(f"At step {iteration}, {lr=}")


def save_image(data: torch.tensor, image_dim: tuple, img_path: str) -> None:
    """
    Creates and stores image given the RGB pixel data, dimension, and save path

    :param data: A tensor of shape (color, num_pixels)
    :param image_dim: Tuple of (width, height)
    :param img_path: String
    :return: None
    """

    # Remember, data is the output of the forward pass, of shape (color, num_pixels)
    # Goal is to reshape it into typical RGB format of shape (num_pixels, color)
    data = data.permute(-1, -2).tolist()
    data = [tuple([round(i) for i in e]) for e in data]  # Convert it into expected format for PIL

    image = create_image(data, image_dim)
    image.save(img_path)



class PEApproximation(nn.Module):
    """
    A PyTorch model for approximating image using PE

    Defines a neural network using PE terms to approximate image through parameterized learning coefficients and biases
    for each RGB color channel and patches. Designed to work with pixel coordinates and output RGB values in forward pass
    """

    def __init__(self, image_dim: tuple, num_terms: int = 5):
        """
        Initialize model

        :param image_dim: Image Dimension
        :param num_terms: Number of PE terms to use for image approximation
        """

        super().__init__()

        self.image_dim = image_dim
        self.num_terms = num_terms

        # Calculate the number of patches
        num_patches = (image_dim[0] // patch_size) * (image_dim[1] // patch_size)
        print(f"With Image Dimension={image_dim}, Patch_Size={patch_size}, There are {num_patches} total patches!")

        # Random Initialization for weights of shape (# of Patches, RBG, PolynomialExp Terms, x-y coord)
        self.coefficients = nn.Parameter(torch.randn(num_patches, 3, num_terms, 2))
        nn.init.xavier_uniform_(self.coefficients)

        # Bias term of shape (3) for each color RGB
        noise_lvl = 255 if start_noisy else 1
        self.bias = nn.Parameter(torch.randn(num_patches, 3) * noise_lvl)
        # nn.init.zeros_(self.bias)


    def forward(self, pix_coord: torch.tensor) -> torch.tensor:
        """
        Forward pass

        :param pix_coord: Initialized Pixel Coordinates from init_data() method
        :return: Returns predicted RGB values as a tensor of shape (color, num_pixels)
        """

        # There are detailed notes of this forward pass implementation within the 'MiscFiles' Directory.
        num_pixels = self.image_dim[0] * self.image_dim[1]
        powers = torch.arange(self.num_terms, dtype=torch.float32, device=device)

        # Pre-calc x and y terms for all pixels
        x_terms = pix_coord[:, 0].unsqueeze(-1) ** powers
        y_terms = pix_coord[:, 1].unsqueeze(-1) ** powers

        # Calculate patch indices for each pixel
        width = self.image_dim[0]
        coordinates = torch.stack([torch.arange(num_pixels) // width, torch.arange(num_pixels) % width], dim=1)
        patch_indices = ((coordinates[:, 0] // patch_size) * (width // patch_size)) + (coordinates[:, 1] // patch_size)

        # Initialize output tensor of expected shape with 0s
        output = torch.zeros(3, num_pixels, device=device)

        # Now calculate the PE terms across all pixels for each color
        for color in range(3):
            output[color] = (
                    torch.sum(self.coefficients[patch_indices, color, :, 0] * x_terms, dim=-1) +
                    torch.sum(self.coefficients[patch_indices, color, :, 1] * y_terms, dim=-1) +
                    self.bias[patch_indices, color]
            )

        return output


# Init data and move to device
image_dimensions, pixel_coords, target_data = init_data()
pixel_coords = pixel_coords.to(device)
target_data = target_data.to(device)

# Save the target image into specified directory first
save_image(target_data.permute(-2, -1), image_dimensions, f"{save_image_dir}Target.png")


model = PEApproximation(image_dim=image_dimensions, num_terms=num_terms).to(device)
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters())


num_params = sum(p.numel() for p in model.parameters())
if num_params < 1000:
    print(f"There are {num_params} parameters in this model")
elif 1000 <= num_params <= 1e6:
    print(f"There are {num_params/1000:.2f}K parameters in this model")
else:
    print(f"There are {num_params/1e6:.2f}M parameters in this model")


start = time.time()
prev_loss = 0
print("Starting")
image_index = 0
initial_loss = -1
for step in range(training_iterations):
    update_lr(step)
    y_pred = model(pixel_coords)
    loss = criterion(y_pred, target_data)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


    if step % eval_interval == 0 or step == training_iterations - 1:

        save_image(y_pred, image_dimensions, f"{save_image_dir}image{image_index}.png")
        print(f"Currently at step={step}   |   loss={loss.item():.2f}   |   time={time.time() - start:.1f}s   |   Loss_decrease={'N/A-' if prev_loss == 0 else round((1 - (loss.item()/prev_loss)) * 100, 2)}%")
        prev_loss = loss.item()
        start = time.time()
        image_index += 1

    if step == 0:
        initial_loss = loss.item()


print(f"|{initial_loss=:.1f}   |   final_loss={loss.item():.1f}   |   Final/Initial-Ratio={loss.item()/initial_loss:.4f}")


