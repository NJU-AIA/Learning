import numpy as np
import json
#y=F(sigma(wi,xi)+b),F为激活函数

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    if x>0:
        return x
    else:
        return 0

def load_data(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
        X = np.array(json_data['x'])
        X = (X - np.mean(X,axis=0)) / np.std(X, axis=0)
        y = np.array(json_data['y'])
        return X,y

def load_ckpt(ckpt_file_name:str):
    with open(ckpt_file_name, "r") as f:
        json_data = json.load(f)
        weights = json_data["weights"]
        return weights

class Layer:
    def __init__(self,input_dim,output_dim,func):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.func=func
        self.weights=np.random.randn(self.input_dim+1,self.output_dim) #加的1是为了1*bi


    def forward(self,x):
        arr=[]
        for vector in x:
            arr.append(np.append(vector,1))
        x=np.array(arr)
        return self.func(x @ self.weights)


    # def backward(self, delta): We'll do this next time!


class MLP:
    def __init__(self,shapes):
        self.layers=[]
        for i in range(len(shapes)-1):
            self.layers.append(Layer(shapes[i],shapes[i+1],sigmoid))

    def forward(self,x):
        for layer in self.layers:
            x=layer.forward(x)
        return x

    # def backward(self, y): We will do this next time!
    # def train(self, train_set, loss='mse', epochs=100, lr=0.3): We will do this next time!

    def evaluate(self, X, y):
        pre=self.forward(X)
        total_correct=0
        for i in range(len(pre)):
            predi=pre[i][0]
            if (np.abs(predi-y[i])<0.5):
                total_correct+=1
        print(pre)
        return total_correct/len(pre)


def main():
    mlp=MLP([2,64,1])
    weights=load_ckpt(r"C:\Users\admin\Downloads\quad.ckpt")
    test_x,test_y=load_data(r"C:\Users\admin\Downloads\data.json")
    for i in range(len(mlp.layers)):
        mlp.layers[i].weights=np.array(weights[i])
        mlp.layers[i].weights=mlp.layers[i].weights.T
    print(mlp.evaluate(test_x,test_y))

main()







