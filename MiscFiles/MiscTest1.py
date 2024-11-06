import torch
from PIL import Image


a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a.T)
print(a.permute(1, 0))

exit()
def create_image(data, image_dim):
    image = Image.new("RGB", image_dim)
    image.putdata(data)
    return image


def save_image(data: torch.tensor, image_dim, img_path: str):
    # Remember, data is the output of the forward pass, of shape (color, num_pixels)
    # Goal is to reshape it into typical RGB format of shape (num_pixels, color)
    data = data.permute(-1, -2).tolist()
    data = [tuple([round(i) for i in e]) for e in data]

    image = create_image(data, image_dim)
    image.save(img_path)




width, height = 320, 240
num_pixels = width * height
patch_size = 16



coord = []
for ith_pixel in range(num_pixels):
    coord.append([ith_pixel // width, ith_pixel % width])




patch_org = []
for c in coord:
    patch_org.append(((c[0] // patch_size) * (width // patch_size)) + (c[1] // patch_size))

# Last two dims for padding of GB values in RGB
patch_org = [patch_org, [0 for _ in range(len(patch_org))], [0 for _ in range(len(patch_org))]]



patch_org = torch.tensor(patch_org)

# patch_org = patch_org.reshape(height, width, 3).permute(0, 2, 1).reshape(3, -1)


# patch_org = patch_org.tolist()
# patch_org = [tuple([round(i) for i in e]) for e in patch_org]



print(patch_org.shape)
print([width, height])
# for w in range(width):
#     print(patch_org[0][w*height: (w+1)*height])
# print("\n\n\n\n############################################\n\n\n\n")
for h in range(height):
    print(patch_org[0][h*width: (h+1)*width])

exit()

save_image(patch_org, [width, height], "t.png")
