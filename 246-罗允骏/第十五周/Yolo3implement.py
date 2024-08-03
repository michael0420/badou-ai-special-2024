import torch
from torchvision.models.detection import yolov3
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

from typing import Any, Callable, List, Optional, Sequence, Tuple
from collections import OrderedDict





def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image[None]  # add batch dimension


def main():
    # 加载预训练的YOLOv3模型
    model = yolov3(pretrained=True)
    model.eval()

    # 加载并预处理图像
    image = load_image("path_to_your_image.jpg")

    # 使用模型进行预测
    with torch.no_grad():
        prediction = model(image)

        # 显示预测结果
    plt.figure(figsize=(12, 8))
    plt.imshow(Image.open("path_to_your_image.jpg"))
    ax = plt.gca()

    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = prediction[0]["scores"][element].cpu().numpy()
        if score > 0.5:
            x1, y1, x2, y2 = boxes
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            ax.text(x1, y1, f"{score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()