from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
from PIL import Image
import io

# class MyDataset(Dataset):
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         byte_image = self.dataset['image'][idx]
#         image = Image.open(io.BytesIO(byte_image))

#         if self.transform:
#             image = self.transform(image)
        
#         return image, self.dataset['label'][idx]

# def data_load(batch_size = 64):
#     cat_dog_dataset_train = load_dataset("JaronU/cat_dog_25000_for_cnn", split='train')
#     cat_dog_dataset_test = load_dataset("JaronU/cat_dog_25000_for_cnn", split='test')

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     train_set = MyDataset(cat_dog_dataset_train, transform)
#     test_set = MyDataset(cat_dog_dataset_test, transform)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

#     return train_loader, test_loader

##################### for tensorflow

def decode_image(image_data, label):
    image = tf.image.decode_jpeg(image_data, channels=3)  
    image = tf.image.resize(image, [224, 224]) 
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label

def load_and_prepare_dataset(split='train'):
    dataset = load_dataset("JaronU/cat_dog_25000_for_cnn", split=split)
    images = tf.convert_to_tensor(dataset['image'])
    labels = tf.convert_to_tensor(dataset['label'])
    labels = to_categorical(labels, num_classes=2)
    
    tf_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    tf_dataset = tf_dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    return tf_dataset

def data_load(batch_size = 64):
    train_dataset = load_and_prepare_dataset('train')
    test_dataset = load_and_prepare_dataset('test')

    train_loader = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_loader = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = data_load()
    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    for x, y in test_loader:
        print(x.shape, y.shape)
        break
    print('data_load test passed')