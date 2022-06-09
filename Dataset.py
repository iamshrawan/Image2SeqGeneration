from cProfile import label
import os
import json
from PIL import Image
from tensorboard import program
import torch
from torchvision import transforms
from torch.utils.data import Dataset

def load_json(json_path):
    with open(json_path, 'r') as j:
        content = json.load(j)
    return content

class Image2NodeDataset(Dataset):
    def __init__(self, im_json, label_json):
        self.images = load_json(im_json)
        self.labels = load_json(label_json)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image_256 = transforms.functional.resize(image, 256, transforms.InterpolationMode.BICUBIC)
        image_t = self.transform(image_256)

        #label
        filename = os.path.split(self.images[idx])[1]
        object_name = '_'.join(filename.split('_')[:2]) 
        sequence = torch.tensor(self.labels[object_name])
        ip_op = sequence[:-1]
        label = sequence[1:]
        return {'image':image_t, 'inp_op':ip_op, 'label': label, 'program_len': len(sequence) - 2}