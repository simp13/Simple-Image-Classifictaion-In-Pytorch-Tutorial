from torchvision import transforms,datasets
from PIL import Image
import torch
import config
import glob
import os

from sklearn import preprocessing
import numpy as np 


class ClassificationDataset:
    def __init__(self,image_paths,targets,transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,item):
        image_path = self.image_paths[item]
        target = self.targets[item]
        image = Image.open(image_path).convert('RGB') # PIL image
        
        if self.transform is not None:
            image = self.transform(image)
        return {
            "images": image,
            "targets": torch.tensor(target,dtype=torch.long),
        }


if __name__ == "__main__":
    
    transform = transforms.Compose([
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.45820624,0.43722707,0.39191988),(0.23130463,0.22692703,0.22379072))
    ])

    label_encoder = preprocessing.LabelEncoder()

    image_paths = glob.glob(os.path.join(config.DATA_DIR,"**/*.*"),recursive=True)
    targets = [x.split("/")[-2] for x in image_paths]
    label_encoded = label_encoder.fit_transform(targets)
    dataset = ClassificationDataset(image_paths,label_encoded,transform)
    print(np.unique(label_encoded))
    print(dataset[0]['images'].size())  
    print(dataset[0]['targets'])
    print(label_encoder.inverse_transform([dataset[0]['targets'].numpy()])[0]) 
