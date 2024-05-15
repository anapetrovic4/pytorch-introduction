import torch
import numpy as np

# Create tensor from numPy ndarray
ndarray = np.array([0,1,2])
t = torch.from_numpy(ndarray)

print(t)

# Inspect attributes of tensor
print(t.shape)
print(t.dtype)
print(t.device)

# Create tensor from list
t = torch.tensor([0,1,2])
print(t)

# Create multidimensional tensor
ndarray = np.array([[0,1,2], [3,4,5]])
t = torch.from_numpy(ndarray)
print(t)

# Create tensor from another tensor (New tensor inherits the characteristics of the initial one)
new_t = torch.rand_like(t,dtype = torch.float) # rand_like returns values [0,1], so we have to overwrite the data type to float
print(new_t)

# Create tensor from the expected shape
my_shape = (3, 3)
rand_t = torch.rand(my_shape)
print(rand_t)

# Tensor operations: slicing, transposing, mutliplying
zeros_tensor = torch.zeros((2,3))
print(zeros_tensor)

# Indexing & slicing
print(zeros_tensor[1])
print(zeros_tensor[:, 0])

# Transpose
transposed = zeros_tensor.T
print(transposed)

# Multiply
ones_tensor = torch.ones(3, 3)
product = torch.matmul(zeros_tensor, ones_tensor)
print(product)

# Loading data
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.MNIST(root='.', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='.', train=False, download=True, transform=ToTensor())

print(training_data[0])
print(training_data.classes)

# Visualize data
figure = plt.figure(figsize=(8,8))
cols, rows = 5,5

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()

# DataLoader - iterate over the dataset in mini batches and shuffle the data while training th emodels
from torch.utils.data import DataLoader

loaded_train = DataLoader(training_data, batch_size=64, shuffle=True)
loaded_test = DataLoader(test_data, batch_size=64, shuffle=True)

# Neural networks - torch.nn module, network is written as a class that inherits from nn.Module
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # Transform to one dimension data
        self.linear_relu_stack = nn.Sequential( # container that creates a sequence of layers
            nn.Linear(28*28, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x): # called when the model is executed
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Instantiate the model
model = NeuralNetwork()
print(model)

# Train the NN

# Set a loss function, such as cross entropy
loss_function = nn.CrossEntropyLoss()

# Set an optimization algorithm - adjust the model during the training process in order to minimize the error measured by the loss function, such as stochastic gradient descent algorithm
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # lr = the speed at which the model's parameters will be updated during each iteration in training

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}")

epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(loaded_train, model, loss_function, optimizer)
    test(loaded_test, model, loss_function)
print("Done!")

torch.save(model, "model.pth")
model = torch.load("model.pth")