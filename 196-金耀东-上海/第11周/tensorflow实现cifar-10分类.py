from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import time , random , threading

# 定义超参数
IMG_SHAPE = [32, 32, 3] # 训练图片shape
DEFAULT_DTYPE = tf.float32 # 参数类型统一为float32
NUM_EPOCHS = 30 # 训练代数
BATCH_SIZE = 512 # 批次大小
W_L2_LOSS = 0.01 # l2_loss权重系数

class NeuralNetwork:
    def __init__(self):
        self.sess = None
        self.x_place = None
        self.y_place = None
        self.y_hat = None
        self.loss = None
        self.optimizer = None

    def _weight_and_bias(self, in_dim, out_dim, k_size=[]):
        '''生成weight与bias'''
        w_shape , b_shape = k_size + [in_dim, out_dim] , [1, out_dim]
        w = tf.Variable( tf.truncated_normal(shape=w_shape, stddev=0.1), dtype=DEFAULT_DTYPE, trainable=True )
        b = tf.Variable( tf.zeros(shape=b_shape), dtype=DEFAULT_DTYPE, trainable=True )
        return w , b

    def _next_batch(self, x, y, batch_size):
        '''读取小批量数据'''
        num_examples = len(y)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            choice = indices[i: min(i + batch_size, num_examples)]
            yield x[choice], y[choice]

    def compile(self):
        # 定义placeholder用于存放输入数据
        self.x_place = tf.placeholder(dtype=tf.float32)
        self.y_place = tf.placeholder(dtype=tf.float32)

        # 第1层:conv1 , input_shape = [batch_size, 32, 32, 3]
        filter_l1 , bias_l1 = self._weight_and_bias(in_dim=3, out_dim=32, k_size=[3,3])
        conv_l1 = tf.nn.conv2d(input=self.x_place, filter=filter_l1, strides=[1, 1, 1, 1], padding="SAME")
        relu_l1 = tf.nn.relu(conv_l1 + bias_l1)
        max_pool_l1 = tf.nn.max_pool(value=relu_l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 第2层:conv2 , input_shape = [batch_size, 16, 16, 32]
        filter_l2 ,  bias_l2 = self._weight_and_bias( in_dim=32, out_dim=64, k_size=[3,3])
        conv_l2 = tf.nn.conv2d(input=max_pool_l1, filter=filter_l2, strides=[1, 1, 1, 1], padding="SAME")
        relu_l2 = tf.nn.relu(conv_l2 + bias_l2)
        max_pool_l2 = tf.nn.max_pool(value=relu_l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 第3层:conv2 , input_shape = [batch_size, 8, 8, 64]
        filter_l3 , bias_l3 = self._weight_and_bias(in_dim=64, out_dim=128, k_size=[3, 3])
        conv_l3 = tf.nn.conv2d(input=max_pool_l2, filter=filter_l3, strides=[1, 1, 1, 1], padding="SAME")
        relu_l3 = tf.nn.relu(conv_l3 + bias_l3)
        max_pool_l3 = tf.nn.max_pool(value=relu_l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 铺平第3层输出数据
        out_l3_dim = 4 * 4 * 128
        out_l3 = tf.reshape(max_pool_l3, shape=[-1, out_l3_dim])

        # 第4层:fc1 , input_shape = [batch_size, 4 * 4 * 128]
        weight_l4 , bias_l4 = self._weight_and_bias(in_dim=out_l3_dim, out_dim=128)
        fc_l4 = tf.matmul(out_l3, weight_l4) + bias_l4
        relu_l4 = tf.nn.relu(fc_l4)

        # 第5层:fc2 , input_shape = [batch_size, 128]
        weight_l5 , bias_l5 = self._weight_and_bias(in_dim=128, out_dim=32)
        fc_l5 = tf.matmul(relu_l4, weight_l5) + bias_l5
        relu_l5 = tf.nn.relu(fc_l5)

        # 第6层:fc3 , input_shape = [batch_size, 64]
        weight_l6 , bias_l6 = self._weight_and_bias(in_dim=32, out_dim=10)
        self.y_hat = tf.matmul(relu_l5, weight_l6) + bias_l6

        # 损失函数：交叉熵 + 卷积核l2范数
        cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y_place, logits=self.y_hat)
        l2_loss = tf.nn.l2_loss(filter_l1) + tf.nn.l2_loss(filter_l2) + tf.nn.l2_loss(filter_l3)
        self.loss = cross_entropy_loss + W_L2_LOSS * l2_loss

        # 优化函数
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def fit(self, x, y, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        # 创建session
        self.sess = tf.Session()

        # 初始化全局参数
        self.sess.run( tf.global_variables_initializer() )

        # 开始训练
        time_start = time.time()
        for epoch in range(num_epochs):
            for i, (x_batch, y_batch) in enumerate(self._next_batch(x, y, batch_size=batch_size)):
                self.sess.run(self.optimizer, feed_dict={self.x_place:x_batch, self.y_place:y_batch})
                if i % 10 == 0:
                    runtime = time.time() - time_start
                    loss , accuracy = self.evalue(x_batch, y_batch)
                    print(f"epoch:{epoch+1} | runtime:{runtime}s | loss:{loss} | accuracy:{accuracy}")
            runtime = time.time() - time_start
            loss, accuracy = self.evalue(x_test, y_test)
            print(f"**** finished epoch {epoch + 1} [ runtime:{runtime}s | loss:{loss} | accuracy:{accuracy} ] ****")

    def evalue(self, x, y):
        '''模型评价函数：返回损失（loss）与准确类（accuracy）'''
        y_pred = self.sess.run(self.y_hat, feed_dict={self.x_place:x})
        loss = self.sess.run(self.loss, feed_dict={self.x_place:x, self.y_place:y})
        accuracy = sum( y_pred.argmax(axis=1) == y.argmax(axis=1) )/ y.shape[0]
        return loss, accuracy

    def predict(self, x):
        '''预测函数：返回预测标签'''
        y_pred = self.sess.run(tf.nn.softmax(self.y_hat), feed_dict={self.x_place:x})
        labels_pred = y_pred.argmax(axis=1)
        return labels_pred

    def close(self):
        '''关闭模型'''
        self.sess.close()



if __name__ == "__main__":
    # 加载CIFAR-10数据集
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 对数据进行预处理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 将分类标签转换为one-hot分类向量
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # 创建模型
    model = NeuralNetwork()

    # 编译模型
    model.compile()

    # 训练模型
    model.fit(x_train, y_train)

    # 评估模型
    loss , accuracy = model.evalue(x_test, y_test)
    print(f"evalue: [ loss:{loss} | accuracy:{accuracy} ]")

    # 关闭模型
    model.close()
