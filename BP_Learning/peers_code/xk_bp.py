# %%
import numpy as np
import json
import random
# TODO some imports
from tqdm import tqdm

# %%
def sigmoid(x):
    return 1/(1+np.exp(-x))
def d_sigmoid(y):
    return y*(1-y)
def relu(x):
    return np.maximum(0,x)
def d_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
def load_data(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
        X = np.array(json_data['x'])
        X = (X - np.mean(X,axis=0)) / np.std(X, axis=0)
        y = np.array(json_data['y'])
        return X,y

# %%
class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(output_size, input_size+1)
        if self.activation == sigmoid:
            self.d_activation = d_sigmoid
        elif self.activation == relu:
            self.d_activation = d_relu
        # TODO param delta
        self.delta = np.zeros(output_size)
    def forward(self, x):
        self.x = np.append(x, 1)
        self.z = self.weights@self.x
        self.y = self.activation(self.z)
        return self.y
    # TODO def backward
    def backward(self, delta):
        self.delta = delta
        return delta*self.d_activation(self.y)@self.weights
    # TODO def update
    def update(self, lr):
        self.weights -= lr*np.outer(self.delta*self.d_activation(self.y), self.x)
    


# %%
class MLP:
    def __init__(self,layer_shapes):
        self.layers = []
        for i in range(len(layer_shapes)-1):
            self.layers.append(Layer(layer_shapes[i],layer_shapes[i+1],sigmoid))
    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def load_ckpt(self, ckpt_file_name):
        with open(ckpt_file_name, 'r') as f:
            json_data = json.load(f)
        weights = json_data['weights']
        assert len(weights) == len(self.layers)
        for index, layer in enumerate(self.layers):
            layer.weights = np.array(weights[index])
    def save_ckpt(self, ckpt_file_name):
        data = {
            'weights': [layer.weights.tolist() for layer in self.layers]
        }
        print(data)
        json_data = json.dumps(data)
        with open(ckpt_file_name, 'w') as f:
            f.write(json_data)
    def evaluate(self, X, ANS_Y):
        acc = 0
        for index, x in enumerate(X):
            pred_y = self.forward(x)[0]
            if np.abs(pred_y - ANS_Y[index]) < 0.5: acc += 1
        return acc/len(ANS_Y)
    # TODO def backward
    def backward(self, delta):
        for layer in reversed(self.layers):
            # delta = layer.backward(delta)
            if layer == self.layers[-1]:
                delta = layer.backward(delta)
            else:
                delta = layer.backward(delta[:-1])
    # TODO def update(self, lr):
    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)
    # TODO def train
    def train(self, X, Y, lr):
        for index, x in enumerate(X):
            pred_y = self.forward(x)[0]
            delta = pred_y - Y[index]
            self.backward(delta)
            self.update(lr)        
        

# %%
lr = 0.001
X, Y = load_data('data.json')
model = MLP([2, 64, 1])
for i in tqdm(range(500)):
    model.train(X, Y, lr)
    # if i % 100 == 0:
        # print(model.evaluate(X, Y))
    
print(model.evaluate(X, Y))


