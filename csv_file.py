import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = r"C:\Users\vaish\OneDrive\Desktop\ML PROJECT\train_data.csv"
df = pd.read_csv(csv_file)

# Initialize lists for features and labels
X = []
y = []

# Loop through each row in the DataFrame to load and preprocess the images
for index, row in df.iterrows():
    img = cv2.imread(row['image_path'])
    img = cv2.resize(img, (64, 64))  # Resize image
    img = img.flatten()  # Flatten image to 1D array
    X.append(img)
    y.append(row['label'])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a MLP (Multi-layer Perceptron) classifier
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Two hidden layers for more complexity
    max_iter=300,  # Increase the number of epochs
    solver='adam',  # Adam optimizer
    learning_rate_init=0.001,  # Small learning rate
    verbose=True  # Show progress during training
)

# Store losses for each epoch
losses = []

# Train the model and track loss at each epoch
for epoch in range(10):  # 300 epochs for smoother curve
    model.fit(X_train, y_train)
    losses.append(model.loss_)

# Plot the loss curve
plt.plot(range(1, 301), losses, marker='o', linestyle='-', color='b')  # Smooth line and markers
plt.title("Loss Curve for 300 Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
