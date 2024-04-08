import numpy as np
import os
import imageio
from PIL import Image
import json


def load_blender_data(basedir, split='train', factor=1, read_offline=True):
    with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
        meta = json.load(fp)

    poses = []
    images = []
    image_paths = []

    for i, frame in enumerate(meta['frames']):
        img_path = os.path.abspath(os.path.join(basedir, frame['file_path'] + '.png'))
        poses.append(np.array(frame['transform_matrix']))
        image_paths.append(img_path)

        if read_offline:
            img = imageio.imread(img_path)
            W, H = img.shape[:2]
            if factor > 1:
                img = Image.fromarray(img).resize((W//factor, H//factor))
            images.append((np.array(img) / 255.).astype(np.float32))
        elif i == 0:
            img = imageio.imread(img_path)
            W, H = img.shape[:2]
            if factor > 1:
                img = Image.fromarray(img).resize((W//factor, H//factor))
            images.append((np.array(img) / 255.).astype(np.float32))

    poses = np.array(poses).astype(np.float32)
    images = np.array(images).astype(np.float32)

    H, W = images[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    return images, poses, [H, W, focal], image_paths


# def load_blender_frame_data(basedir, frame_i, factor=1, read_offline=True, split="train"):
#     with open(os.path.join(basedir, f"frame_{frame_i}.json"), "r") as fp:
#         meta = json.load(fp)

#     poses = []
#     images = []
#     image_paths = []

#     for i, frame in enumerate(meta["camera_data"]):
#         img_path = os.path.abspath(os.path.join(basedir, frame["file_path"]))
#         poses.append(np.array(frame["transform_matrix"]))
#         image_paths.append(img_path)

#         if read_offline:
#             img = imageio.imread(img_path)
#             W, H = img.shape[:2]
#             if factor > 1:
#                 img = Image.fromarray(img).resize((W // factor, H // factor))
#             images.append((np.array(img) / 255.0).astype(np.float32))
#         elif i == 0:
#             img = imageio.imread(img_path)
#             W, H = img.shape[:2]
#             if factor > 1:
#                 img = Image.fromarray(img).resize((W // factor, H // factor))
#             images.append((np.array(img) / 255.0).astype(np.float32))

#     poses = np.array(poses).astype(np.float32)
#     images = np.array(images).astype(np.float32)

#     H, W = images[0].shape[:2]
#     camera_angle_x = float(meta["camera_angle_x"])
#     focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

#     return images, poses, [H, W, focal], image_paths


# a = load_blender_frame_data("C:\\Users\\leois\\OneDrive\\Documents\\SFU\SFUTERM2\\733\papr\\data\\butterfly\\frame_000","000")
# a = load_blender_data("C:\\Users\\leois\\OneDrive\\Documents\\SFU\\SFUTERM2\\733\\papr\\data\\nerf_synthetic\\chair")
# print(a)
