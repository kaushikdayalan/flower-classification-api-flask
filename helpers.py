import io
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn

def get_model():
    checkpoint = "trained_model.pt"
    classifier = models.resnet18(pretrained = True)
    classifier.fc = nn.Linear(512, 5)
    classifier.load_state_dict(torch.load(checkpoint, map_location="cpu"),strict = False)
    classifier.eval()
    return classifier

def get_tensor_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(size=(224,224)),transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    image = my_transforms(image)
    image = image.unsqueeze(0)
    return image
