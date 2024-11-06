from PIL import Image
import torch


def get_image_data(image_path, patch_size):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_dim = image.size

    if image_dim[0] != image_dim[1]:
        print(f"Given image has dimensions={image_dim}, which is not a square image and would leave to artifacts\n")
        make_square = input("Adjust image into square? [Y/N]: ")
        while make_square.lower() not in ["y", "n"]:
            print("Invalid Input")
            make_square = input("Adjust image into square? [Y/N]: ")
        make_square = make_square.lower() == "y"

        if make_square:
            print()
            print(f"Current ratio of width/height = {(image_dim[0]/image_dim[1]):.3f}\n"
                  f"Cropping is recommended over compression for ratios deviating more than ~10%")
            use_cropping = input("Crop image? [Y/N]: ")
            while use_cropping.lower() not in ["y", "n"]:
                print("Invalid Input")
                use_cropping = input("Crop image? [Y/N]: ")

            image_dim = (min(image_dim), min(image_dim))
            use_cropping = use_cropping.lower() == "y"
            if use_cropping:
                print("Now cropping image...")
                print("Trimming is applied bidirectionally")
                image = crop_square_image(image)
            else:
                print("Now compressing image...")
                image = image.resize(image_dim)

            print()


    return padded_data(image, image_dim, patch_size)


def crop_square_image(image):
    # Default crops both sides until square, assumes central focal point which is majority of case. Adjust as needed

    width, height = image.size
    crop_width = width > height

    odd = abs(width - height) % 2  # 1 if Odd, 0 if Even
    crop_pixels = abs(width - height) // 2  # Number of pixels to crop in both direction

    # If True, it means width is greater than height, so we crop left and right sides
    if crop_width:
        # Crop method takes in tuple of 4 coordinate integers, representing upper-left and lower-right
        # Note that last two coordinate for lower-right is exclusive whereas upper-left is inclusive
        image = image.crop((crop_pixels, 0, width-crop_pixels-odd, height))
    else:  # Else, crop top and bottom
        image = image.crop((0, crop_pixels, width, height-crop_pixels-odd))

    # Shouldn't trigger
    assert image.size[0] == image.size[1], "Cropped image should be square"
    return image


def padded_data(image, image_dim, patch_size):
    data = list(image.getdata())  # Original shape (pixel, color)
    pad_x = (patch_size - (image_dim[0] % patch_size)) % patch_size
    pad_y = (patch_size - (image_dim[1] % patch_size)) % patch_size


    data = torch.tensor(data, dtype=torch.float32).reshape(image_dim[1], image_dim[0], 3)

    # Permute the tensor then pad accordingly. Using white at the moment for a more visible contrast
    data = torch.nn.functional.pad(data.permute(2, 0, 1), (0, pad_x, 0, pad_y), mode='constant', value=255)
    # Permute the padded tensor back into its original format
    data = data.permute(1, 2, 0)

    data = data.reshape(-1, 3)  # Return it back to original format of (pixel, color) shape
    image_dim = (image_dim[0] + pad_x, image_dim[1] + pad_y)

    # Return shape tensor(pixel, color)
    return data, image_dim



def create_image(data, image_dim):
    image = Image.new("RGB", image_dim)
    image.putdata(data)
    return image


if __name__ == "__main__":
    # list_data, image_dimension = get_image_data("./Images/Mandelbrot_Fractal_320x240.png", 10)
    # r, g, b = create_RGB_image(list_data, image_dimension)
    # r.save("./Images/red.png")
    # g.save("./Images/green.png")
    # b.save("./Images/blue.png")

    # data, image_dim = get_image_data("./Images/Mandelbrot_Fractal_320x240.png", 100)
    # data = [tuple([int(e) for e in d]) for d in data.tolist()]
    #
    # image = create_image(data, image_dim)
    # image.save("T1.png")

    img_path = "./Images/Star.jpg"
    # img_path = "./colorful.png"
    image = Image.open(img_path)
    # image = image.resize((1600, 1600))
    image = crop_square_image(image)
    image.save("./temp.png")


