"""
Just a quick explanation of how the values for PE was calculated within the forward pass to serve as documentation for myself

* PE = Polynomial Expansion


Original implementation:
------------------------------------------------------
for color in range(3):
    for ith_pixel in range(num_pixels):
        # Calculate the PE for each pixel (pix_coord[i, 0] for x-coordinate, pix_coord[i, 1] for y-coordinate)
        # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
        for jth_term in range(self.num_terms):
            output[color][ith_pixel] += (
                    self.coefficients[color, jth_term, 0] * (pix_coord[ith_pixel, 0] ** jth_term) +  # x-related terms
                    self.coefficients[color, jth_term, 1] * (pix_coord[ith_pixel, 1] ** jth_term)  # y-related terms
            )

Where the 'coefficients' parameter tensor is only a 3d tensor, of shape (num_colors, num_terms, x-y)
Where num_colors=3, num_terms corresponds to number of terms used to calculate the PE values, usually 10, and x-y is of size 2, where 0 represents x and 1 represents y

This uses 3 nested loops, which is extremely inefficient. I purposely made it this way to serve as the basis to build up on.
Creating this first makes it easy to see the step-by-step vectorization and allows me to understand it better.



Next, vectorization of the innermost for-loop, removing 'jth_term' completely.
------------------------------------------------------
powers = torch.arange(self.num_terms, dtype=torch.float32, device=device)
for color in range(3):
    for ith_pixel in range(num_pixels):
        # Calculate the patch index. There's a video that explains the logic for calculating the patch index.
        width = self.image_dim[0]
        coordinate = [ith_pixel//width, ith_pixel % width]
        patch_index = ((coordinate[0]//patch_size) * (width//patch_size)) + (coordinate[1] // patch_size)

        # Calculate the PE for each pixel (pix_coord[i, 0] for x-coordinate, pix_coord[i, 1] for y-coordinate)
        # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
        output[color][ith_pixel] = (
            torch.sum(self.coefficients[patch_index, color, :, 0] * (pix_coord[ith_pixel, 0] ** powers)) +  # Corresponds to X values
            torch.sum(self.coefficients[patch_index, color, :, 1] * (pix_coord[ith_pixel, 1] ** powers)) +  # Corresponds to Y values
            self.bias[patch_index, color]  # The associated bias with this current patch
        )

So instead of going through it one by one, vectorization of the entire portion that represents the PE terms and sum it up, making it much more efficient

Note that Coefficients, likewise bias, is now a 4d tensor. I've added a num_path dimension in the forefront when creating the parameter tensors.
It is now of shape (num_patches, color=3, num_terms, x-y=2)
Where num_patches is the number of 'patches' in the image, determined by the hyperparameter patch_size.
My earliest implementation overlooked this 'block' feature, which resulted in an image of a series of unrecognizable pattern.
Taking a look through, I noticed that it's because I since the exact same coefficients and bias parameters are used for the image,
only the x-y coordinate values serves as variable for resulting predicted values. This results in a pattern rather than an actual image.
Splitting it up into multiple different 'patches' or 'blocks' would render out a much better image.



The next step was removing the outer loop for colors,
------------------------------------------------------
for ith_pixel in range(num_pixels):
    # Calculate the patch index. There's a video that explains the logic for calculating the patch index.
    width = self.image_dim[0]
    coordinate = [ith_pixel//width, ith_pixel % width]
    patch_index = ((coordinate[0]//patch_size) * (width//patch_size)) + (coordinate[1] // patch_size)

    # Calculate the PE for each pixel (pix_coord[i, 0] for x-coordinate, pix_coord[i, 1] for y-coordinate)
    # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
    output[:, ith_pixel] = (
        torch.sum(self.coefficients[patch_index, :, :, 0] * (pix_coord[ith_pixel, 0] ** powers), dim=-1) +  # Corresponds to X values
        torch.sum(self.coefficients[patch_index, :, :, 1] * (pix_coord[ith_pixel, 1] ** powers), dim=-1) +  # Corresponds to Y values
        self.bias[patch_index, :]  # The associated bias with this current patch
    )

This is fairly straightforward. Just replace all instances of 'color' with ':', due to how tensor-indices work
One thing to note is the added dim=-1 parameter. Because the goal is to get 3 individual values, one for each color, this
added parameter will sum across the num_terms dimension, leaving behind a tensor of shape 3, rather than 1 if dim parameter is not attended to



Finally the last optimization, the most crucial one with the removal of ith_pixel loop rather than color
------------------------------------------------------
    def forward(self, pix_coord: torch.tensor):
        num_pixels = self.image_dim[0] * self.image_dim[1]
        powers = torch.arange(self.num_terms, dtype=torch.float32, device=device)

        x_terms = pix_coord[:, 0].unsqueeze(-1) ** powers  # Shape: (num_pixels, num_terms)
        y_terms = pix_coord[:, 1].unsqueeze(-1) ** powers  # Shape: (num_pixels, num_terms)

        # Calculate patch indices for each pixel
        width = self.image_dim[0]
        coordinates = torch.stack([torch.arange(num_pixels) // width, torch.arange(num_pixels) % width], dim=1)
        patch_indices = ((coordinates[:, 0] // patch_size) * (width // patch_size)) + (coordinates[:, 1] // patch_size)

        output = torch.zeros(3, num_pixels, device=device)

        for color in range(3):
            output[color] = (
                    torch.sum(self.coefficients[patch_indices, color, :, 0] * x_terms, dim=-1) +
                    torch.sum(self.coefficients[patch_indices, color, :, 1] * y_terms, dim=-1) +
                    self.bias[patch_indices, color]
            )

        return output

Here, the x and y terms are precomputed.
For the process for calculating is the following:

Taking x_terms as the example, pix_coord[:, 0] would grab all rows, column 0, corresponding to all x-values
This is of shape (num_pixels), then using .unsqueeze to add an additional dimension after it to get shape (num_pixels, 1)
'powers' variable is a tensor from range 0 to num_terms, therefore its shape is (num_terms).
Due to broadcasting, once the current value of x_terms is raised to 'powers', the resulting x_terms variable is now
of shape (num_pixels, num_terms)
The same applies for y_terms

Next is the calculation of patch_indices. This is relatively straightforward. Recall the the original calculation
for each individual coordinate for patch index is

    coordinate = [ith_pixel//width, ith_pixel % width]

Here it is modified to accommodate vectorization using arange

    coordinates = torch.stack([torch.arange(num_pixels) // width, torch.arange(num_pixels) % width], dim=1)

Originally coordinate is a list of 2 elements, representing the [row, column] index, here it will now be  shape=(num_pixels, 2)
Then that would be used to calculate patch indices, a tensor of shape (num_pixels)

The rest are fairly self-explanatory.

# This final optimization is quite a large one. The previous two are small-scaled optimization, a rough estimate would be around
# 1-5 times faster (A very rough estimate). A fairly good amount. However this one is a major improvement, quite literally over a thousand times faster
# after some testing. Which is amazing, and also makes sense. Vectorization of both 'color' and 'jth_term' loop although improves
# the efficiency, but it is small, where color is 3, RGB, and number of PE terms are usually from 5-20. However the crux is at
# num_pixels, as a photo can have ranging from tens of thousands to millions of pixels, depending on the resolution. This magnitude
# completely dwarfs the size of other two, making it the emphasis.












Here's the original forward pass, saving it just in case

    def forward_old(self, pix_coord: torch.tensor):
        # First, calculate the number of pixels in an image of each color via width * height
        num_pixels = self.image_dim[0] * self.image_dim[1]

        # # Initialize the output tensor of each image's color, fill with corresponding bias. Size = (num_pixels)
        # red_bias = torch.full((num_pixels,), self.bias[0].item())
        # green_bias = torch.full((num_pixels,), self.bias[1].item())
        # blue_bias = torch.full((num_pixels,), self.bias[2].item())
        # # Stack it to get (RGB, num_pixels)
        # output = torch.stack((red_bias, green_bias, blue_bias))

        output = torch.zeros(3, num_pixels, device=device)

        # for color in range(3):
        #     for ith_pixel in range(num_pixels):
        #         # Calculate the PE for each pixel (pix_coord[i, 0] for x-coordinate, pix_coord[i, 1] for y-coordinate)
        #         # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
        #         for jth_term in range(self.num_terms):
        #             output[color][ith_pixel] += (
        #                     self.coefficients[color, jth_term, 0] * (pix_coord[ith_pixel, 0] ** jth_term) +  # x-related terms
        #                     self.coefficients[color, jth_term, 1] * (pix_coord[ith_pixel, 1] ** jth_term)  # y-related terms
        #             )

        powers = torch.arange(self.num_terms, dtype=torch.float32, device=device)
        for color in range(3):
            for ith_pixel in range(num_pixels):
                # Calculate the patch index. There's a video that explains the logic for calculating the patch index.
                width = self.image_dim[0]
                coordinate = [ith_pixel//width, ith_pixel % width]
                patch_index = ((coordinate[0]//patch_size) * (width//patch_size)) + (coordinate[1] // patch_size)

                # Calculate the PE for each pixel (pix_coord[i, 0] for x-coordinate, pix_coord[i, 1] for y-coordinate)
                # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
                output[color][ith_pixel] = (
                    torch.sum(self.coefficients[patch_index, color, :, 0] * (pix_coord[ith_pixel, 0] ** powers)) +  # Corresponds to X values
                    torch.sum(self.coefficients[patch_index, color, :, 1] * (pix_coord[ith_pixel, 1] ** powers)) +  # Corresponds to Y values
                    self.bias[patch_index, color]  # The associated bias with this current patch
                )


        # Return the output, shape=(3, num_pixels)
        return output


"""