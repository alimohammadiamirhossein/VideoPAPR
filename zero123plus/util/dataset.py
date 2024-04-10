import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


class Zero123Dataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        if transform is None:
            transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # Add more transformations as needed
            ])

        super().__init__(root, transform=transform)

    def load_images_from_directory(self, directory):
        gts = {}
        inputs = {}
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                frame_num = filename.split('_')[0]
                frame_type = filename.split('_')[1].split('.')[0]
                filepath = os.path.join(directory, filename)
                if int(frame_num) >= 5: # it is hard coded for now # number of frames
                    continue
                image = Image.open(filepath)
                image = image.convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                if image is not None:
                    if frame_type == 'input':
                        inputs[frame_num] = image
                    elif frame_type == 'gt':
                        gts[frame_num] = image
        gts = [gts[k] for k in sorted(gts.keys(), key=lambda x: int(x))]
        inputs = [inputs[k] for k in sorted(inputs.keys(), key=lambda x: int(x))]
        gts = torch.stack(gts)
        inputs = torch.stack(inputs)
        return inputs, gts

    def __getitem__(self, index):
        path, _ = self.samples[index]
        path = '/'.join(path.split('/')[:-1])
        inputs, gt = self.load_images_from_directory(path)
        return inputs, gt


# # Define your data directory
# data_dir = '/localhome/aaa324/Generative Models/VideoPAPR/zero123plus/dataset'
#
# # Define transformations (if needed)
#
#
# # Create custom dataset
# custom_dataset = Zero123Dataset(root=data_dir, transform=None)
#
# # Create data loader
# batch_size = 1
# data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
#
# # Iterate over the data loader
# for batch in data_loader:
#     inputs, gt = batch
#     print(inputs.shape, 12)
#     # Process your batch of images as needed
