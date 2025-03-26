# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

In this task, we there is a dataset that contains multiple input features (independent variables) and a continuous target variable (dependent variable). The model should learn the relationships between these variables to make accurate predictions on new data.

To achieve this, we should:

Preprocess the dataset – Handle missing values, normalize features, and split the dataset into training and testing sets.

Build the neural network model – Design and implement a regression model using a deep learning framework like TensorFlow/Keras.

Train the model – Use the training dataset to optimize model parameters through backpropagation and gradient descent.

Evaluate the model – Assess the model’s performance using evaluation metrics such as Mean Squared Error (MSE) and R-squared (R²).

Fine-tune the model – Optimize hyperparameters such as learning rate, number of layers, and neurons per layer to improve accuracy.

## Neural Network Model

![{CDD1914A-FB1B-4490-8D76-FFAEC4CF106D}](https://github.com/user-attachments/assets/fbfff882-fade-409f-9e1d-114d20f5781e)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:SREE NIVEDITAA SARAVANAN
### Register Number:212223230213
```python
class NeuralNet(nn.Module):
    class NeuralNet(nn.Module):
    def __init__(self): # Corrected the typo here: __init__
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.001)



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train), y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

![{B5A745C4-745A-4326-AF17-4E949D16EDD4}](https://github.com/user-attachments/assets/868ab533-6a08-44e1-956e-c755d1697ffe)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-26 104052](https://github.com/user-attachments/assets/db33960b-bd43-4638-b950-54130cfaacd2)


### New Sample Data Prediction

![Screenshot 2025-03-26 104107](https://github.com/user-attachments/assets/ce073545-6cdd-407f-87ba-681b5b63f173)


## RESULT

Thus , the program has been successfully executed.
