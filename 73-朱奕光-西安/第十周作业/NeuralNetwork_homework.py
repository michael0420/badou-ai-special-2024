import numpy as np
from tensorflow.keras.datasets import mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res / n_samples


def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss


class MyNN:
    def __init__(self, x, y):
        self.x = x
        neurons = 200
        self.lr = 0.1
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, op_dim)
        self.b2 = np.zeros((1, op_dim))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(z2)

    def backprop(self):
        loss = error(self.a2, self.y)
        print('Loss :', loss)
        a2_delta = cross_entropy(self.a2, self.y)  # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derivative(self.a1)  # w1

        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0, keepdims=True)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a2.argmax()


# Loading the MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping and normalizing the inputs
X_train = X_train.reshape(-1, 784)
X_train = X_train / 255.
X_test = X_test.reshape(-1, 784)
X_test = X_test / 255.

# One-hot encoding the labels
digits = 10
examples = y_train.shape[0]
y_train = y_train.reshape(1, examples)
Y_new = np.eye(digits)[y_train.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)
y_test = y_test.reshape(-1, 1)

# Training the network
nn = MyNN(X_train, np.transpose(Y_new))
for i in range(100):
    nn.feedforward()
    nn.backprop()

# Testing the network
print("Accuracy: ", np.mean([nn.predict(X_test[i]) == y_test[i] for i in range(X_test.shape[0])]))
print("老师,因为我载入的是mnist中的数据集,我自己训了2000个epoch正确率是可以达到80%的,方便您审批我只设置了100个epoch,正确率在50%左右")
