import numpy as np
import matplotlib.pyplot as plt
import struct

def load_images(file):
    with open(file, 'rb') as f:
        magic, size = struct.unpack('>ii', f.read(8))
        rows, cols = struct.unpack('>ii', f.read(8))
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape((size, rows*cols))
    
    return images/255.0

def load_labels(file):
    with open(file, 'rb') as f:
        magic, size = struct.unpack('>ii', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    
    return labels

X_train = load_images('assets/MNIST/raw/train-images-idx3-ubyte')
y_train = load_labels('assets/MNIST/raw/train-labels-idx1-ubyte')
X_test = load_images('assets/MNIST/raw/t10k-images-idx3-ubyte')
y_test = load_labels('assets/MNIST/raw/t10k-labels-idx1-ubyte')

# 显示一张图片
def show_image(index):
    image = X_train[index].reshape(28,28)  # 第 index 张图片
    label = y_train[index]  # 对应的标签
    print(f"Label: {label}")
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # 初始化网络结构和超参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        return MLP.sigmoid(z) * (1 - MLP.sigmoid(z))

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray):
        # 前向传播
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, y):
        # 计算损失（交叉熵损失）
        m = y.shape[0]
        log_probs = -np.log(self.A2[range(m), y])
        loss = np.sum(log_probs) / m
        return loss

    def backward(self, X, y):
        # 反向传播
        m = X.shape[0]
        y_onehot = np.zeros_like(self.A2)
        y_onehot[range(m), y] = 1

        dZ2= self.A2 - y_onehot
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 更新权重和偏置
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                # 前向传播
                self.forward(X_batch)

                # 计算损失
                loss = self.compute_loss(y_batch)

                # 反向传播
                self.backward(X_batch, y_batch)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        # 预测
        A2 = self.forward(X)
        return np.argmax(A2, axis=1)

    def evaluate(self, X, y):
        # 测试集评估准确率
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    # 保存模型
    def save_model(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Model saved to {filename}")

    # 加载模型
    def load_model(self, filename):
        # 没有找到文件不加载，输出但是不报错，直接返回
        try:
            data = np.load(filename)
        except FileNotFoundError:
            print(f"Model file {filename} not found.")
            return

        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Model loaded from {filename}")


input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 20
batch_size = 64

mlp = MLP(input_size, hidden_size, output_size, learning_rate)


# 加载模型
# mlp.load_model('mlp_model1.npz')

# 训练模型
mlp.train(X_train, y_train, epochs, batch_size)

train_accuracy = mlp.evaluate(X_train, y_train)
test_accuracy = mlp.evaluate(X_test, y_test)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
mlp.save_model('mlp_model1.npz')

# 测试模型
id = 3
show_image(id)
print(mlp.predict(X_train[id]))