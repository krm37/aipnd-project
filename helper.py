from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torchvision
import argparse
from PIL import Image
import numpy as np
from torch import nn

def get_data_loader(data_dir:str):
    
    train_dir=data_dir+"train"
    valid_dir=data_dir+"valid"
    
    train_transforms=transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    valid_transforms=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])


    train_images = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_images = datasets.ImageFolder(valid_dir,transform=valid_transforms)

    trainloader = DataLoader(train_images,batch_size=64,shuffle=True)
    validloader = DataLoader(valid_images,batch_size=64,shuffle=True)
    
    return trainloader,validloader,train_images.class_to_idx


def get_train_input_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,  help='path to the folder of flower images') 
    parser.add_argument('--save_dir', type=str, default='saved_models', help='path to save the model') 
    parser.add_argument('--arch', type=str, default='densenet169', help='CNN model architecture') 
    parser.add_argument('--learning_rate', type=float, default= 0.01, help='learning rate for the model') 
    parser.add_argument('--hidden_units', type=str, default= "[512]", help='hidden unit size for the model') 
    parser.add_argument('--epochs', type=int , default= 20, help='hidden unit size for the model') 
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
        

    args = parser.parse_args()

    return args

def get_prediction_input_args():
    parser =argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='path to input image') 
    parser.add_argument('checkpoint', type=str, help='path to saved model') 
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to the file containing category names')
    parser.add_argument('--topk', type=int, default= 5, help='topk class for prediction')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    return args


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image_path)
    transformer = transforms.Compose([transforms.Resize(256), 
                                         transforms.CenterCrop(224), 
                                         transforms.ToTensor(), 
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])
    
    pil_image = transformer(pil_image)

    return np.array(pil_image)

def create_classifier(input_size, hidden_units, output_size):
    layers = []
    prev_size = input_size

    for hidden_size in hidden_units:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        prev_size = hidden_size

    layers.append(nn.Linear(prev_size, output_size))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

model_dict = {
    "densenet169": torchvision.models.densenet169(pretrained=True),
    "resnet50": torchvision.models.resnet50(pretrained=True),
    "vgg19": torchvision.models.vgg19(pretrained=True)
}
