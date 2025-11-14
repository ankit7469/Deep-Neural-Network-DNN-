#  Deep Neural Network (DNN) — TensorFlow/Keras

This project demonstrates a **Deep Neural Network (DNN)** built using **TensorFlow & Keras**.  
The model includes multiple dense layers, **Batch Normalization**, and **Dropout** to improve training stability and prevent overfitting.

---

## 🚀 Project Highlights---

- Implemented a **multi-layer DNN**
- Used **ReLU** activation for hidden layers
- Used **sigmoid** activation for output (binary classification)
- Applied **Batch Normalization** to stabilize learning
- Added **Dropout** to reduce overfitting
- Visualized **training loss & accuracy**
- Made predictions on test inputs

---

##  What I Learned -----

### ✔ Deep Neural Network (DNN)
- A neural network with **multiple hidden layers**
- Learns complex, non-linear relationships
- Useful for classification and regression tasks

### ✔ Batch Normalization
- Normalizes layer outputs  
- Helps model converge faster  
- Reduces internal covariate shift  

### ✔ Dropout
- Randomly drops neurons during training  
- Prevents overfitting  
- Makes network more robust  

### ✔ Adam Optimizer
- Adaptive learning rate  
- Works well for most deep learning tasks  
- Faster convergence  

### ✔ Training Visualization
- Loss function decreasing over time  
- Accuracy improving each epoch  

---

## 🧾 Tech Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

---

## 💻 Code Overview

```python
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(16, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(8, activation='relu'),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, y, epochs=300)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(["Loss","Accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training Progress")
plt.grid(True)
