import matplotlib.image as mpimg
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

from alexnet import AlexNet


def print_answer(argmax):
    with open("index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
        print(synset)
    print(synset[argmax])


def get_predict(model, img):
    with torch.no_grad():
        outputs = model(img)
    print(outputs)
    _, predicted_idx = torch.max(outputs, 1)
    print(_, predicted_idx)
    return predicted_idx.item()


if __name__ == '__main__':
    # net = AlexNet()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = net.to(device)
    # model.load_state_dict(torch.load("path_to_model_weights"))
    model = AlexNet()
    model.load_state_dict(torch.load("logs/last1.pth"))
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256x256
        transforms.CenterCrop(224),  # 从中心裁剪出224x224的图像
        transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并缩放到[0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open("./Test.jpg")
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # 增加一个batch维度

    predicted_idx = get_predict(model, image_tensor)
    print_answer(predicted_idx)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
