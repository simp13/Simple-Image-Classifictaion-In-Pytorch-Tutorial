
import os
import glob
import torch
import numpy as np

from torchvision import transforms,datasets

from sklearn import preprocessing
from sklearn import model_selection

import config
import dataset
import engine
from model import SmallNet


def run_training():
    transform = transforms.Compose([
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.45820624,0.43722707,0.39191988),(0.23130463,0.22692703,0.22379072))
    ])

    label_encoder = preprocessing.LabelEncoder()

    image_paths = glob.glob(os.path.join(config.DATA_DIR,"**/*.*"),recursive=True)
    targets = [x.split("/")[-2] for x in image_paths]
    label_encoded = np.array(label_encoder.fit_transform(targets))
    
    (train_images,test_images,train_labels,test_labels) = model_selection.train_test_split(image_paths,label_encoded,test_size=0.2,random_state=0)
    # print(len(train_images))
    # print(len(train_labels))

    train_dataset = dataset.ClassificationDataset(train_images,train_labels,transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
    
    test_dataset = dataset.ClassificationDataset(test_images,test_labels,transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config.BATCH_SIZE,shuffle=False)

    model =SmallNet(num_classes=3)
    model.to(config.DEVICE)

    opt = torch.optim.Adam(model.parameters(),lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.8, patience=5, verbose=True
    )

    for epoch in range(config.EPOCHS):
        
        train_loss = engine.train_fn(model,train_loader,opt)
        val_accuracy,val_loss = engine.eval_fn(model,test_loader)

        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={val_loss} Accuracy={val_accuracy}"
        )
        
        scheduler.step(val_loss)

    print("Saved model...")
    torch.save(model.state_dict(),"./models/weights_latest.pt")    

if __name__ == "__main__":
    run_training()