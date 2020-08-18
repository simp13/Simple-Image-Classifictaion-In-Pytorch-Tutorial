import numpy as np
from model import SmallNet
from PIL import Image 
import matplotlib.pyplot as plt 
from torchvision import transforms
import torch 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,help="Path to input image")
args = vars(ap.parse_args())

transform = transforms.Compose([
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.45820624,0.43722707,0.39191988),(0.23130463,0.22692703,0.22379072))
    ])

labels = ['cat','dog','panda']
path2modelweight = "./models/weights_latest.pt"

model = SmallNet(num_classes=3)
model.load_state_dict(torch.load(path2modelweight))

model.eval()
image = Image.open(args['input'])

with torch.no_grad():
    x = transform(image)
    y,_ = model(x.unsqueeze(0))
    y_out = torch.softmax(y,dim=1).numpy()
    y_out = np.argmax(y_out)
    print("Label->",labels[y_out])
    fig,ax = plt.subplots()
    ax.imshow(image)
    plt.show()


