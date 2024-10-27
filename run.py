import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_network.neural_network import NeuralNetwork


# Read data
df = pd.read_csv('data/winequality-white.csv', header=0, sep=';')

# Normalize X columns
for col in df.columns:
    if col != 'quality':
        df[col] = df[col] / df[col].max()

# Normalize y column
df['quality'] = df['quality'] / 10

n_train = 200
n_test = 50

# Training set
X_train = df.drop(columns=['quality'])[:n_train].values
y_train = df['quality'][:n_train].values.T

# Test set
X_test = df.drop(columns=['quality'])[n_train:n_train+n_test].values
y_test = df['quality'][n_train:n_train+n_test].values.T


nn = NeuralNetwork()

nn.add_layer(num_neurons=X_train.shape[1], input_size=X_train.shape[1])
nn.add_layer(num_neurons=X_train.shape[1], input_size=3)
nn.add_layer(num_neurons=X_train.shape[1], input_size=3)
nn.add_layer(num_neurons=1, input_size=3)

# Train the network
nn.train(X_train, y_train, epochs=5000, learning_rate=0.5)

# Save the model after training
nn.save("models/wine_quality_model.json")

# Load the model (no need to train)
nn.load("models/wine_quality_model.json")

# Predict using the trained network
y_pred = nn.predict(X_test)

# Refactor quality to units
y_pred = np.round(y_pred*10)
y_test = np.round(y_test*10)

print(f"Predictions: {y_pred[:10]}")
print(f"Real: {y_test[:10]}")

plt.plot(nn.loss_list)
plt.show()

plt.plot(y_test[:n_test])
plt.plot(y_pred[:n_test])
plt.show()
