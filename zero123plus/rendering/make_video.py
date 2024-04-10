import cv2
import os


def create_video(image_directory, output_path, type='input', frame_rate=30):
    # Get the list of input images
    input_images = [img for img in os.listdir(image_directory) if img.endswith(f"_{type}.png")]
    input_images.sort()  # Sort the images in numerical order

    # Determine the width and height of the images
    img = cv2.imread(os.path.join(image_directory, input_images[0]))
    height, width, _ = img.shape

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Iterate over the input images and add them to the video
    for image_name in input_images:
        image_path = os.path.join(image_directory, image_name)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()

# Usage example
image_directory = "/localhome/aaa324/Generative Models/Examples/snoopy"
for ftype in ['input', 'gt']:
    output_path = os.path.join(image_directory, f"{ftype}_video.mp4")
    create_video(image_directory, output_path, type=ftype)
