import cv2
import os


def compile_video(img_folder: str, fps: int, output_filename: str, use_step: bool) -> None:
    """
    Creates the video file based on given images

    :param img_folder: Folder that contains the images
    :param fps: Desired FPS of output video
    :param output_filename: Output video name
    :param use_step: Boolean value, set to True if image steps are linearly spaced and later images takes longer to converge
    :return: None
    """

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
        print(image)  # Prints out image filename to track current progress

    video.release()
    print("Video created successfully!")


def get_images(img_folder: str, use_step: bool) -> list[str]:
    """
    Gathers all image paths and returns it as a sorted list based on time created

    :param img_folder: Folder that contains the images
    :param use_step: Boolean value, set to True if image steps are linearly spaced and later images takes longer to converge
    :return:
    """

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

