import numpy as np
import json

def load_data(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
        X = np.array(json_data['x'])
        X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
        y = np.array(json_data['y'])
        return X,y 

def load_ckpt(ckpt_file_name:str):
    with open(ckpt_file_name, "r") as f:
        json_data = json.load(f)
        weights = json_data["weights"]
        return weights
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    if x <= 0:
        return 0
    else:
        return x

class Layer:
    def __init__(self, input_dim, output_dim, function):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.function = function
        self.weights = np.zeros((input_dim + 1, output_dim))

    def forward(self, x):
        arr = []
        for vector in x:
            arr.append(np.append(vector, 1))
        x = np.array(arr)
        return self.function(x @ self.weights)
    

class MLP:
    def __init__(self, shapes:list):
        self.layers = []
        for i in range(len(shapes)-1):
            self.layers.append(Layer(shapes[i], shapes[i+1], sigmoid))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def evaluate(self, x, y):
        pred = self.forward(x)
        total_correct = 0
        for i in range(len(pred)):
            if(np.abs(pred[i][0]) - y[i] < 0.5):
                total_correct += 1
        return total_correct / len(pred)  

def main():
    mlp = MLP([2,64,1])
    test_X, test_y = load_data("./python/MLP/data.json")
    weights = load_ckpt("./python/MLP/quad.ckpt")
    for i in range(len(mlp.layers)):
        mlp.layers[i].weights = np.array(weights[i]).T
    print(mlp.evaluate(test_X, test_y))

main()