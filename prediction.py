import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Get Optimized Thetas
theta = np.load("Data\\model_theta.npy")
### Get Data
data = pd.read_csv("Data\\data.csv")
### Implement Prediction
def Prediction(x,theta):
    prediction = x.dot(theta)
    return prediction

data.drop(columns=["waterfront","country","street","statezip","date"],inplace=True)

data["city"] = data["city"].astype("category").cat.codes

split_index = int(len(data) * 0.8)

for col in data.columns:
    if col != "price":
        data[col] =  (data[col] - data[col].min()) / (data[col].max() - data[col].min()) 

data.insert(1,"base",1)
x = data.iloc[split_index:,1:]
y = data.iloc[split_index:,0]

x_np = x.to_numpy()
y_np = y.to_numpy()

y_predictoin = Prediction(x_np,theta)

y_unscaled = y_predictoin * (y_np.max() - y_np.min()) + y_np.min()

y_y_predic = np.c_[y_np,y_unscaled]

print(y_y_predic[:100])

plt.figure(figsize=(10, 6))

plt.plot(y_np[:100], label='Actual Price', linewidth=2)
plt.plot(y_unscaled[:100], label='Predicted Price', linewidth=2)

plt.title('Actual vs Predicted Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Data\\Images\\my_plot1")
plt.show()

x_axis = list(range(len(y_np)))

plt.scatter(x_axis, y_np, c='blue', label="Actual Values", s=10)
plt.scatter(x_axis, y_unscaled, c='red', label="Prediction Values", s=10)

plt.title("Actual vs Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Price (scaled)")
plt.legend()
plt.grid(True)
plt.savefig("Data\\Images\\my_plot2")
plt.show()
