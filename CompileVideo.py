import cv2
import os



def compile_video(img_folder, fps, output_filename, use_step):
    images = get_images(img_folder, use_step)
    print(f"Using {len(images)} images to create {fps}fps video")

    if not images:
        raise ValueError("No images found in the folder")

    frame = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(img_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
        print(image)

    video.release()
    print("Video created successfully!")


def get_images(img_folder, use_step):
    images = sorted(
        [img for img in os.listdir(img_folder) if img.endswith(".jpg") or img.endswith(".png")],
        key=lambda img: os.path.getmtime(os.path.join(img_folder, img))
    )

    if use_step:
        result = []
        step = 50
        mini_step = 1
        max_mini_step = 5
        increment = 1
        for i in range(0, len(images), step):
            result.extend(images[i: i + step: mini_step])
            mini_step = min(max_mini_step, 4 if mini_step == 1 else mini_step + increment)

        return result

    return images



if __name__ == "__main__":
    image_folder = "./CreatedImages/PE_Images"
    compile_video(image_folder, 30, 'Star.mp4', use_step=True)

