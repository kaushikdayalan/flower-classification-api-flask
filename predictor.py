from helpers import get_model,get_tensor_image
import torch

classes = ["daisy","dandelion","rose","sunflower","tulip"]

model = get_model()

def get_prediction(image_bytes):
    image = get_tensor_image(image_bytes)
    prediction = None
    predicted_values = model(image)
    if prediction is None:
        prediction = predicted_values.data.cpu()
    else:
        prediction = torch.cat((prediction, prediction.data.cpu()), dim=0)
    _, preds = torch.max(prediction, 1)
    preds = preds.numpy()
    return (classes[preds[0]])