import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

from utils import PATH_MODELS, MODEL_VERSION, C_NUM, GESTURE_CLASSES, log_decorator, FONT
import global_vars

# input image dimensions
img_rows, img_cols = 200, 200

## Number of output classes (change it accordingly)
## eg: In my case I wanted to predict 4 types of gestures (Ok, Peace, Punch, Stop)
## NOTE: If you change this then dont forget to change Labels accordingly
nb_classes = 6


# img transformer

# mobileNet
# data_transform = transforms.Compose([transforms.Resize(256),
#                                      transforms.CenterCrop(224),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                      ])

# my naive NN
data_transform = transforms.Compose([transforms.Resize(64),
#     transforms.CenterCrop(224),
    transforms.Grayscale(),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5]), # 1 chanel
])


class NaiveNN(nn.Module):
    def __init__(self, num_classes=C_NUM):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def snap_loader(img, cuda=True):
    """input video stream snap, returns cuda tensor"""
    img = Image.fromarray(img)
    img = data_transform(img).float()
    # img = Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    return img.cuda() if cuda else img


def initialization(cuda=True):
    # init torch
    device=None
    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

@log_decorator
def load_model(pretrained=True):
    if pretrained:
        try:
            # model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=False)
            model = NaiveNN()
            model.load_state_dict(torch.load(PATH_MODELS + MODEL_VERSION))
            return model
        except Exception as e:
            print("Error: can't fetch trained model.", e)
            exit(1)
    else:
        print("ERROR: retrain function not provided yet.")
        exit(1)


def trainModel(model):
    pass


def update():
    h = 450//2
    y = 30//2
    w = 45//2
    plot = np.zeros((512//2,512//2,3), np.uint8)
    # print("update plot")
    for key in global_vars.jsonarray.keys():
        mul = global_vars.jsonarray[key]
        plot=cv2.line(plot, (0, y), (int(h * mul), y), (255, 200, 180), w)
        plot=cv2.putText(plot, GESTURE_CLASSES[key], (0,y + 5), FONT, 1, (200, 255, 255), 2)
        y = y + w + 30//2

    return plot


def predict_gesture(model, img, cuda=True,verbose=True):
    """
    predict gesture, meanwhile update the JsonArray, which is used to save probabilities.
    :param model: neural network
    :param img: ROI of a frame
    :return: prediction result
    """
    img = snap_loader(img, cuda)
    output = model(img)
    _, preds_tensor = torch.max(output, 1)
    # calculate probabilities
    # probs = F.softmax(output, dim=1).detach().numpy()  # numpy shape like [1,6], mobileNet
    probs = torch.exp(output)
    # update jsonarray
    tmp = {i: probs[0][i].item() for i in range(C_NUM)}
    global_vars.jsonarray = tmp
    if verbose:
        print("Guess gesture: %s" % GESTURE_CLASSES[int(preds_tensor)])
    return int(preds_tensor)
