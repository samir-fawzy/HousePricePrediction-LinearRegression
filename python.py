import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Get data
path = "Data\\data.csv"
data = pd.read_csv(path)

data.drop(columns=["waterfront","country","street","statezip","date"],inplace=True)

data["city"] = data["city"].astype("category").cat.codes


for col in data.columns:
    if col != "Price":
        data[col] =  (data[col] - data[col].min()) / (data[col].max() - data[col].min()) 

x = data.iloc[:,1:]
y = data.iloc[:,0]

split_index = int(len(data) * 0.8)

x_training = x.iloc[:split_index]
y_training = y.iloc[:split_index]

x_testing = x.iloc[split_index:]
y_testing = y.iloc[split_index:]

x_training_np = np.c_[np.ones(x_training.shape[0]),x_training.to_numpy()]
y_training_np = y_training.to_numpy().reshape(-1,1)

x_testing_np = np.c_[np.ones(x_testing.shape[0]),x_testing.to_numpy()]
y_testing_np = y_testing.to_numpy().reshape(-1,1)

theta = np.zeros((x_training_np.shape[1],1))

def Prediction(x,theta):
    result = x.dot(theta)
    return result

def ComputeCost(x,y,theta):
    prediction = Prediction(x,theta)
    sumSquareError = np.power(prediction - y,2).sum()
    m = len(y)
    return sumSquareError / (m * 2)

def GradientDescent(x,y,iters = 20000,alpha=0.1):
    m = len(y)
    theta = np.zeros((x.shape[1],1))
    for i in range(iters):
        gradient = (1 / m) * x.T.dot((x.dot(theta) - y))
        theta = theta - alpha * gradient
        cost = ComputeCost(x,y,theta)
        print(f"Cost at iteration{i} = {cost}")
    return theta

opt_theta = GradientDescent(x_training_np,y_training_np)

print(opt_theta)

np.save("Data\\model_theta.npy",opt_theta)



