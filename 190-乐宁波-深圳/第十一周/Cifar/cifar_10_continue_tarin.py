from keras.models import load_model
import cifar_10_data

# 加载已经训练好的模型
model = load_model('D:\\basic_cnn_model.h5')

# 数据目录
data_dir = 'cifar_data/cifar-10-batches-bin'
batch_size = 100

# 加载训练和测试数据
train_dataset = cifar_10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
test_dataset = cifar_10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 继续训练模型
history = model.fit(train_dataset, epochs=15, validation_data=test_dataset)

# 保存重新训练的模型
model.save('D:\\basic_cnn_model.h5')

# 打印训练结果
print("继续训练完成。")
