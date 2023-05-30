# A gentle introduction to deep learning in PyTorch

# Table of Contents
1. [Introduction to PyTorch](#introduction-to-pytorch)
2. [Introduction to Deep Learning in PyTorch](#deep-learning-basics-in-pytorch)
3. [Convolutional Neural Network basics](#introduction-to-cnns)
4. [Recurrent Neural Network basics](#introduction-to-rnns)
5. [Generative Adversarial Networks](#generative-adversarial-networks-gans)
6. [Variational Autoencoders](#variational-autoencoders-vaes)
7. [Reinforcement Learning](#reinforcement-learning)

## Introduction to PyTorch
### What is a Tensor? 
A tensor, in the context of PyTorch, is a multi-dimensional array used to represent data. It's similar to arrays in Python or ndarrays in NumPy, but with additional features that make it suitable for deep learning applications.

### Installation of PyTorch
Before we start coding, you need to install PyTorch. Assuming that you have Python installed, you can install PyTorch by running the following command in your terminal:

```python
!pip install torch torchvision torchaudio
```

This command installs not only PyTorch (`torch`), but also `torchvision` and `torchaudio`, which provides datasets and model architectures for computer vision and audio processing.

### Getting Started with Tensors
To start with PyTorch, we need to import it:

```python
import torch
```

You can create a tensor with a specific data type using `torch.tensor()`. Here is an example:

```python
t1 = torch.tensor([1, 2, 3])
print(t1)
```

In this code, we've created a 1-dimensional tensor. You can also create a 2-dimensional tensor like this:

```python
t2 = torch.tensor([[1, 2], [3, 4]])
print(t2)
```

You can check the shape of a tensor (i.e., its dimensions) using the `.shape` attribute:

```python
print(t2.shape)
```

This tells you that `t2` is a 2x2 tensor (a matrix with 2 rows and 2 columns).

#### Mathematical Operations
Now let's dive into some mathematical operations that we can perform on tensors.

##### Addition:
You can add two tensors of the same shape:

```python
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
t3 = t1 + t2

print(t3)
```

##### Multiplication:
You can also multiply two tensors:

```python
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
t3 = t1 * t2

print(t3)
```

This is element-wise multiplication, not matrix multiplication.

##### Matrix Multiplication:
For matrix multiplication, we can use `torch.matmul()`:

```python
t1 = torch.tensor([[1, 2], [4, 3]])
t2 = torch.tensor([[3, 4], [2, 1]])
t3 = torch.matmul(t1, t2)

print(t3)
```

### Indexing, Slicing, Joining, and Mutation
Let's explore how to manipulate tensors in PyTorch. The operations we will focus on are indexing, slicing, joining, and mutating tensors.

#### Indexing and Slicing
Like in Python and many other languages, indexing in PyTorch starts at 0. Let's create a 1-D tensor and access its elements:

```python
t1 = torch.tensor([1, 2, 3, 4, 5])

# accessing the first element
print(t1[0])

# accessing the last element
print(t1[-1])
```

Slicing works similarly to Python's list slicing. You can select a range of elements in a tensor using the syntax `[start:end]`:

```python
print(t1[1:3])  # prints elements at index 1 and 2
```

Indexing and slicing for multi-dimensional tensors work similarly, just with additional dimensions:

```python
t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# accessing a single element
print(t2[1, 2])  # prints element at row 1, column 2

# slicing
print(t2[1:, :2])  # prints all rows starting from 1, and columns up to 2
```

#### Joining Tensors
You can join multiple tensors into one. The most common way to do this is by using `torch.cat()`:

```python
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])

# concatenating along the 0th dimension
t3 = torch.cat((t1, t2), dim=0)

print(t3)
```
You can also concatenate 2-D tensors along either dimension:

```python
t1 = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])

t3 = torch.cat((t1, t2), dim=0)  # concatenating along rows
print(t3)

t4 = torch.cat((t1, t2), dim=1)  # concatenating along columns
print(t4)
```

#### Mutation
Finally, let's look at how to modify a tensor. You can change an element of a tensor by accessing it and assigning a new value:

```python
t1 = torch.tensor([1, 2, 3])

t1[1] = 7  # changing the second element

print(t1)
```

You can also modify a slice of a tensor:

```python
t1 = torch.tensor([1, 2, 3, 4, 5])

t1[1:4] = torch.tensor([7, 8, 9])  # changing elements at index 1, 2, and 3

print(t1)
```

Remember that PyTorch tensors are mutable, meaning you can change their content without creating a new tensor.
### CUDA support, Moving computations to GPU
Deep learning models can be computationally intensive, and running them on a CPU can be slow. That's where Graphics Processing Units (GPUs) come in. GPUs are designed for performing large numbers of computations simultaneously, making them ideal for deep learning.
CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia for general computing on its own GPUs. PyTorch has built-in support for CUDA, which allows it to seamlessly perform computations on a GPU, if one is available.

#### Checking for CUDA Availability
Before we can move computations to a GPU, we first need to check whether CUDA is available. We can do this using `torch.cuda.is_available()`:

```python
print(torch.cuda.is_available())
```

This will output `True` if CUDA is available and `False` otherwise.

#### Moving Tensors to GPU
You can create a tensor on a GPU directly, or move a tensor from CPU to GPU.

To create a tensor on a GPU:

```python
t1 = torch.tensor([1, 2, 3], device='cuda:0')
print(t1)
```

Here, `cuda:0' refers to the first GPU. If you have multiple GPUs and you want to create a tensor on the second GPU, you would use 'cuda:1', and so on.

To move a tensor from CPU to GPU:

```python
t2 = torch.tensor([4, 5, 6])
t2 = t2.to('cuda')

print(t2)
```

#### Performing Computations on GPU
Once a tensor is on a GPU, any operations performed on it are carried out on the GPU:

```python
t1 = torch.tensor([1, 2, 3], device='cuda')
t2 = torch.tensor([4, 5, 6], device='cuda')

t3 = t1 + t2  # this computation is performed on the GPU

print(t3)
```

Remember to ensure that all tensors involved in an operation are on the same device. If they're not, you'll need to move them first. Note that while using a GPU can greatly speed up computations, it also comes with its own challenges, such as managing GPU memory. However, these are problems for a more advanced tutorial.

### Automatic Differentiation with Autograd
When training neural networks, we often need to compute the gradients of loss functions with respect to the model's parameters, which are then used to update the parameters. This can be done using a technique called backpropagation.
Luckily, PyTorch provides a package called autograd for automatic differentiation, which simplifies the computation of backward passes. This is an essential tool for neural networks, and PyTorch makes it quite easy to use.

#### Tensors and Gradients
In PyTorch, you can tell an autograd-compatible tensor to remember its operations for backpropagation using `.requires_grad_(True)`:

```python
t1 = torch.tensor([1., 2., 3.], requires_grad=True)
print(t1)
```

This tensor `t1` now has the ability to keep track of every operation involving it. Now, let's perform some operations:

```python
t2 = t1 * 2
t3 = t2.mean()
print(t3)
```

Here, `t3` is created as a result of operations involving `t1`, so it has a `grad_fn`.

#### Backward Propagation
To perform backpropagation, we call `.backward()` on the tensor. This computes the gradient of `t3` with respect to `t1`:

```python
t3.backward()
```

After calling `.backward()`, the gradients are stored in the `.grad` attribute of the original tensor (`t1`):

```python
print(t1.grad)
```

The gradient stored in `t1.grad` is the partial derivative of `t3` with respect to `t1`. It's important to note that if `t1`'s shape doesn't match with `t3`, PyTorch won't be able to directly compute the gradients. That's why when we are dealing with scalar loss in a neural network, we don't have to provide any arguments to `backward()`. Also, PyTorch accumulates the gradient in the `.grad` attribute, which means that it doesn't overwrite the previous values but instead adds the newly computed gradient. To clear the gradients for a new operation, you need to call `.grad.zero_()`.

### PyTorch Libraries: Torchtext, torchvision, torchaudio
PyTorch is not just a deep learning library; it also has several companion libraries that provide additional functionality, such as handling specific types of data and performing various common operations. Today, we'll discuss three of these libraries: `torchtext`, `torchvision`, and `torchaudio`.

#### Torchtext

`torchtext` is a library for handling text data. It provides tools for creating datasets, handling tokenization, and managing vocabularies. It also includes several common datasets and pre-trained embeddings (like Word2Vec or GloVe). Here is a simple example of using `torchtext`:

```python
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torchtext.data import TabularDataset
from collections import Counter

# Tokenization
tokenizer = get_tokenizer('spacy', language='en')

# Define fields
def tokenize_text(text):
    return [token.lower() for token in tokenizer(text)]

# Define dataset
dataset = TabularDataset(path="mydata.csv", format='csv', fields=[("text", tokenize_text), ("label", LABEL)])

# Build vocab
counter = Counter()
for (text, label) in dataset:
    counter.update(tokenize_text(text))
vocab = Vocab(counter, vectors="glove.6B.100d")
```

#### Torchvision
`torchvision` is a library for handling image data. It includes tools for transforming images, common image datasets (like CIFAR10, MNIST), and pre-trained models (like ResNet, VGG). Here is a simple example of using `torchvision`:

```python
from torchvision import datasets, transforms

# Define a transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load a dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# torchvision also provides pre-trained models
from torchvision import models
model = models.resnet50(pretrained=True)
```

#### Torchaudio
`torchaudio` is a library for handling audio data. It provides tools for loading and saving audio data, transformations (like spectrograms, mel-scaling), and some common audio datasets. Here is a simple example of using `torchaudio`:

##### Audio I/O backend
`torchaudio` needs an audio I/O backend called SoundFile to read and write audio files. Install it using pip:

```python
!pip install SoundFile
```
Let's look at some common stuff that `torchaudio` is used for:
```python
import torchaudio

# Load an audio file
waveform, sample_rate = torchaudio.load('audio.wav') # Replace with a WAV file on your local drive

# Apply a transform
transform = torchaudio.transforms.MFCC(sample_rate=sample_rate)
mfcc = transform(waveform)

# torchaudio also includes some common datasets
yesno_data = torchaudio.datasets.YESNO('.', download=True)
```

Let's now turn our attention to the basics of deep learning

## Deep Learning Basics in PyTorch
### Building a Basic Feedforward Neural Network
A feedforward neural network, also known as a multi-layer perceptron (MLP), is a basic type of neural network where information moves in only one direction—it is fed forward from the input layer, through any number of hidden layers, to the output layer. There are no cycles or loops in the network.

#### Building a Feedforward Neural Network with PyTorch
With PyTorch, we can easily build a feedforward neural network by defining a class that inherits from `torch.nn.Module`. Let's start with a simple network that has one input layer, one hidden layer, and one output layer.

First, we need to import the `torch` module and the `torch.nn` module, which contains the building blocks needed to build neural networks:

```python
import torch
import torch.nn as nn
```

Next, we define our neural network:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 10)  # 10 input neurons, 10 output neurons
        self.fc2 = nn.Linear(10, 1)   # 1 output neuron

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation function on first layer
        x = self.fc2(x)              # Output layer
        return x
```

Here's a breakdown of the code:

* We create a class `Net` that inherits from `nn.Module`.
* In the constructor (`__init__`), we define two fully connected layers (`fc1` and `fc2`). `nn.Linear` is used to apply a linear transformation to the incoming data: y = x*A^T + b.
* `forward(x)` is a method that implements the forward propagation of the input `x`. `torch.relu` is the Rectified Linear Unit (ReLU) activation function that is applied elementwise.

We can now create an instance of the neural network:

```python
net = Net()
print(net)
```

Now that we have defined the network architecture, the next step would be to define a loss function and an optimizer, pass the input data through the network, calculate the loss, backpropagate the error, and update the weights. These steps are typically performed in a training loop, which we will discuss in a bit.

### Training a Feedforward Neural Network
Training a neural network involves iterating over a dataset of input samples and adjusting the weights of the network to minimize a loss function.

Let's continue with the feedforward neural network we built in the previous lesson. We'll use a simple regression problem where the task is to predict a single target value from 10 input features. We'll create an artificial dataset using `torch.randn()` for this purpose.

#### Preparing the Data
Let's generate a simple synthetic dataset:

```python
input_features = torch.randn(100, 10)  # 100 samples, 10 features each
target_values = torch.randn(100, 1)    # 100 target values
```

This creates two tensors: `input_features` containing 100 samples each with 10 features, and `target_values` containing 100 corresponding target values. The values are randomly generated.

#### Defining a Loss Function and an Optimizer
We'll use Mean Squared Error (MSE) as our loss function—it's a common choice for regression problems. For optimization, we'll use Stochastic Gradient Descent (SGD). PyTorch provides these as `torch.nn.MSELoss` and `torch.optim.SGD` respectively:

```python
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # net is our network
```

The optimizer adjusts the weights of the network based on the gradients computed during backpropagation.

#### Training the Network
Training involves a loop over epochs (iterations over the whole dataset). In each epoch, we:
1. Run the forward pass to get the outputs from the input features,
2. Calculate the loss between the outputs and the target values,
3. Backpropagate the error through the network,
4. Use the optimizer to adjust the weights.

```python
for epoch in range(100):  # Number of times to loop over the dataset
    # Forward pass
    outputs = net(input_features)
    
    # Compute loss
    loss = loss_fn(outputs, target_values)
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Perform backward pass (backpropagation)
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
```

After running this code, you'll see the loss decreasing over epochs, which means our model is learning to fit the input features to the target values.

### Evaluating a Feedforward Neural Network
After training a neural network, it's crucial to evaluate its performance on a separate test dataset that the model has never seen during training. This helps to determine if the model is generalizing well or just memorizing the training data (overfitting).
We will continue from the feedforward neural network model we trained in the previous lesson.

#### Preparing the Test Data
Similar to the training data, we'll create synthetic test data:

```python
test_features = torch.randn(50, 10)  # 50 samples, 10 features each
test_targets = torch.randn(50, 1)    # 50 target values
```

#### Evaluating the Model
Now, let's evaluate our model on this test data. While doing so, we'll switch off PyTorch's autograd engine using `torch.no_grad()`. This reduces memory usage and speeds up computations but is unable to perform backpropagation.

```python
with torch.no_grad():  # Switch off gradients
    test_outputs = net(test_features)
```

#### Calculating the Loss
To calculate the loss between our model's predictions and the true values:

```python
test_loss = loss_fn(test_outputs, test_targets)
print(f'Test Loss: {test_loss.item()}')
```

This will print the test loss value. In a perfect model, this would be 0, but in practice it's always higher. Ideally, this test loss should be close to the training loss. If the test loss is much higher, it might be a sign that the model is overfitting.

### Backpropagation and Computational Graphs
Backpropagation is a critical process in training deep learning models. It is a method used to calculate the gradient of the loss function with respect to each weight in the model, which is then used to update the weights. Backpropagation is made possible in PyTorch through the construction of computational graphs during the forward pass. In this lesson, we will discuss computational graphs and backpropagation in more detail.

#### What is a Computational Graph?
A computational graph is a directed graph where the nodes correspond to operations or variables. Variables can be input tensors, parameters (such as weights and biases), or intermediate values. Operations can be any mathematical operations, such as addition and multiplication, or more complex ones like activation functions.

When a forward pass is made in a neural network, a computational graph is dynamically created in PyTorch. This graph is used to compute the gradients during the backward pass, which is a fundamental step in optimizing the network.

#### Backpropagation and Computational Graphs
Let's consider a simple example to understand how PyTorch constructs a computational graph and how backpropagation is performed. Let's take two tensors a and b, which are parameters of the model and require gradient computation, and a simple equation f = (a * b) + b.

```python
import torch

# Initialize tensors
a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([3.], requires_grad=True)

# Compute outputs
f = (a * b) + b

# Compute gradients
f.backward()

# Print gradients
print(a.grad)  # df/da
print(b.grad)  # df/db
```

In this example, a computational graph is created in the following way:

1. Nodes are created for the tensors a and b.
2. When f = (a * b) + b is computed, nodes are created for the operations * and +, and for the output f. Edges are created to connect these operations.
3. During the backward pass (`f.backward()`), the gradients of the output with respect to each tensor that has `requires_grad=True` are computed using this computational graph.
4. The returned gradients are the derivatives of f with respect to a and b and are stored in the `.grad` property of the respective tensors.

In more complex models like deep neural networks, the computational graph can be far more intricate, but the idea holds. During backpropagation, the gradients of the loss with respect to the weights are computed and used to update the weights.

### Building Neural Networks using nn.Module
We've already touched on the usage of `nn.Module` in PyTorch when building our simple feedforward network, but it's worth diving deeper into how it works and why it's such a fundamental part of structuring models in PyTorch.

`nn.Module` is a base class for all neural network modules in PyTorch. Your models should also subclass this class, which allows you to utilize powerful PyTorch functionalities. Modules can contain other Modules, allowing to nest them in a tree structure.

#### Building a Network with nn.Module
Let's expand our basic feedforward network into a multi-layer one using `nn.Module`:

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input layer to hidden layer
        self.hidden = nn.Linear(10, 20) 

        # Hidden layer to output layer
        self.output = nn.Linear(20, 1)  

        # Activation function
        self.relu = nn.ReLU()             

    def forward(self, x):
        x = self.hidden(x)  # Pass data through hidden layer
        x = self.relu(x)    # Apply activation function
        x = self.output(x)  # Pass data through output layer
        return x
```

In the `Net` class defined above, we first define the layers of our network in the constructor `__init__()`. We have an input to hidden layer connection (`self.hidden`), an activation function (`self.relu`), and a hidden to output layer connection (`self.output`).

Then, in the `forward` function, we define the path that the input tensor will take through the network.

#### Why nn.Module?
* Parameter management: Once layers are defined as members of `nn.Module` (in our example, `self.hidden` and `self.output`), all parameters of these layers are registered in the module. You can access all parameters of your network using the `parameters()` or `named_parameters()` methods.
* Device management: You can recursively move all parameters of the network to a device (CPU or GPU) by calling `.to(device)` on the network. This saves you the trouble of moving each tensor individually.
* Easy to use with optimizers: You can pass the parameters of a network (retrieved through `net.parameters()`) to an optimizer, and it will take care of the parameter updates for you.
* Helper methods and functionalities: There are many other functionalities provided by `nn.Module` such as the ability to save and load the model, ability to set the network in training or evaluation mode (`net.train()`, `net.eval()`).

### Debugging PyTorch models and using torchviz to visualize networks
Debugging PyTorch models and visualizing the computational graph of your model is an essential part of understanding and developing deep learning models. PyTorch does not come with a built-in tool for this, but a popular third-party library, `torchviz`, can be used to create a visualization of your model.

#### Debugging PyTorch Models
Debugging PyTorch models can be done in the same way as debugging any Python program, using Python's built-in debugging tools. The most popular is `pdb`, the Python debugger. You can insert breakpoints into your code using `pdb.set_trace()`, which allows you to step through your code and inspect variables.

For example, during the forward pass of your model:

```python
import pdb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        pdb.set_trace()  # Set breakpoint here
        x = self.fc(x)
        return x
```

When the forward method is called, the program will pause at `pdb.set_trace()`. You can then print the values of variables, step through the code, and more. To continue execution, type c and hit enter.

#### Visualizing Networks with torchviz
`torchviz` is a PyTorch specific package that creates directed graphs of PyTorch execution traces and computational graphs. It's handy to understand the operations and tensor transformations that take place during the forward and backward pass.

To use `torchviz`, you first need to install it, and also be sure to download the installer from https://graphviz.org/download/:

```python
!pip install torchviz
```

Here is how to use it to visualize the computation graph of a simple model:

```python
import torch
from torch import nn
from torchviz import make_dot

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1,8)

# Forward pass
y = model(x)

# Create a visualization of the forward computation graph
make_dot(y, params=dict(model.named_parameters())).render("rnn_torchviz", format="png")
```

Let's now look at increasing the sophistication of our network through convolution.

## Introduction to CNNs 
Convolutional Neural Networks (CNNs) are a type of deep learning model primarily used for tasks related to image processing, such as image classification, object detection, and more. Unlike regular fully-connected networks, CNNs are designed to automatically and adaptively learn spatial hierarchies of features from the input data. This basically means that CNNs have the ability to understand and learn various levels of abstract features from the input data (like an image) all by themselves.

To illustrate, consider an example of a CNN trained to recognize faces. In the beginning layers of the network, it might learn to detect simple features such as edges and curves. Moving deeper into the network, the subsequent layers might start to recognize combinations of edges and curves that form more complex features like eyes, noses, and mouths. Further down the network, even more complex features like a face can be detected.

So, it's like starting from simple features and gradually building up a 'hierarchy' of increasingly complex features. This is what we refer to as spatial hierarchies. The network does all this without any explicit programming instructing it to learn these features – hence the term "automatically and adaptively".

CNNs are founded on basic building blocks: convolutional and pooling layers.

### Understanding Convolutions and Pooling
#### Convolutional Layer
A convolutional layer applies a series of different image filters, also known as convolutional kernels, to the input image. Each filter is small spatially (along width and height), but extends through the full depth of the input volume. For example, a typical filter on a first layer of a CNN might have size 5x5x3 (i.e., 5 pixels width and height, and 3 because images have depth 3, the color channels).

As the filter slides over the image (or another layer's feature map), it is convolved with that portion of the image, computing the dot product between the entries of the filter and the input image at any position. 

Imagine you're given a large image, and your task is to find a smaller image within that large image. One way you might approach this is by taking your smaller image and sliding it over every possible position in the large image to see where it fits best.

This is essentially what a convolutional layer does. It takes a set of "small images" (filters) and slides them over the input image, checking how well they match. When a filter matches well with a certain region of the image, the convolutional layer will output a high value, indicating that it has found a feature it recognizes.

Here's a simplified example in PyTorch:

```python
# PyTorch has its own conv2d function in the torch.nn.functional module
import torch
from torch import nn
import torch.nn.functional as F

# input is a 1x1 image with a 3x3 matrix of ones
inputs = torch.ones(1, 1, 3, 3)

# filters are a single 2x2 filter
filters = torch.ones(1, 1, 2, 2)

outputs = F.conv2d(inputs, filters, stride=1, padding=0)

print(outputs)
```

### Pooling Layer
The other building block of CNNs is the pooling layer. Pooling layers reduce the dimensions of the data by combining the outputs of neuron clusters at one layer into a single neuron in the next layer. 

Imagine you're given a task to describe a large, complex image with a small number of words. You'll probably pick the most important or distinctive features of the image to describe, effectively summarizing or "downsizing" the information.

This is the concept of pooling. The pooling layer takes a complex input (like the output of a convolutional layer), and summarizes it into a smaller, simpler form. This "summary" retains the most important features while discarding less useful details. This not only helps to reduce computational load and control overfitting, but also provides a form of translation invariance - meaning the network will recognize the same feature even if it's slightly shifted in the image. The most common type of pooling is max pooling, which takes the maximum value in each window of the input.

Here's a simple example of a max pooling operation in PyTorch:

```python
# input is a 1x1 image with a 4x4 matrix
inputs = torch.Tensor([[[[1, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 1]]]])

# MaxPool2d
m = nn.MaxPool2d(2, stride=2)
output = m(inputs)

print(output)
```

### CNN Architecture
A typical CNN architecture is made up of several layers:
* Convolutional Layer: This layer performs a convolutional operation, creating several smaller picture windows to go over the data.
* Non-Linearity (ReLU): After each convolutional layer, it is convention to apply a non-linear layer (or activation function) immediately afterward.
* Pooling or Sub Sampling: This layer is periodically inserted in-between successive convolutional layers.
* Classification (Fully Connected Layer): The last stage in a CNN. It takes all the previous outputs and flattens them into a single vector that can be fed into a fully connected layer for classification purposes.

### Building CNNs in PyTorch
#### A simple CNN
First, let's define a simple CNN architecture. We will build a network that has:

* A convolutional layer with 3 input channels, 6 output channels, a kernel size of 3 (a 3x3 filter), and a stride of 1.
* A ReLU activation function.
* A max pooling operation with 2x2 window and stride 2.
* A second convolutional layer with 6 input channels, 16 output channels, a kernel size of 3, and a stride of 1.
* Another ReLU activation function.
* Another max pooling operation.
* A fully connected (linear) layer that maps the input units to 120 output units.
* Another fully connected layer that maps the previous 120 output units to 84 output units.
* A final fully connected layer that maps the 84 output units to 10 output classes.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 3)  # 3 input channels, 6 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(6, 16, 3)  # 6 input channels, 16 output channels, 3x3 kernel
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

Here, we've defined two methods: `__init__ `and `forward`. The `__init__` method initializes the various layers we need for our network. The `forward` method defines the forward pass of the network. The backward pass (computing gradients) is automatically defined for us using autograd.

Let's do a basic test of our model. First, let's initialize our model:

```python
net = Net()
print(net)
```

Now, let's create a random 3x32x32 input (remember, our network expects 3-channel images):

```python
input = torch.randn(1, 3, 32, 32)
out = net(input)
```

This prints out the 10-dimensional output of the network.

```python
print(out)
```

Let's try zeroing the gradient buffers of all parameters and backprops with random gradients:

```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```

This doesn't do much yet, but in a short bit, we'll see how to backpropagate errors based on a loss function and update the model's weights. This simple implementation isn't intended to achieve meaningful results - it's just to give you a sense of how the model operates.

### Training a CNN, Overfitting and Regularization Techniques
#### Training a CNN
Training a CNN involves several steps:

1. Forward Propagation: In the forward pass, you pass the input data through the network and get the output.
2. Loss Computation: After obtaining the output, you calculate the loss function. The loss function shows how far the network output is from the true output. The aim is to minimize this loss.
3. Backward Propagation: In the backward pass, also known as backpropagation, the gradient of the loss is calculated with respect to the parameters (or weights) of the model, and the parameters are updated in the opposite direction of the gradients to reduce the loss.
4. Optimization: Finally, the weights of the network are updated using the gradients computed during backpropagation.

Here is a simple example of how to train a CNN in PyTorch:

```python
import torch.optim as optim

# create your network
net = Net()

# define a loss function
criterion = nn.CrossEntropyLoss()

# define an optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for each epoch...
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    # for each batch of data...
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

This will train your CNN on the dataset for 2 epochs. Note that this is a simple example for learning purposes; in real-world cases, you would likely need to train for more epochs and also split your dataset into training and validation sets to monitor and prevent overfitting.

#### Overfitting and Regularization Techniques
Overfitting occurs when a neural network model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model.

Here are some common ways to prevent overfitting:

* Data Augmentation: This is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data. Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks.
* Dropout: Dropout is a regularization method where input and recurrent connections to a layer are probabilistically excluded from activation and weight updates while training a network. This has the effect of training a large ensemble of neural network models that share weights.
* Early Stopping: This involves stopping the training process before the learner passes that point where performance on the test dataset starts to degrade.
* Weight Decay: Also known as L2 regularization, weight decay involves updating the loss function to penalize the model in proportion to the size of the model weights.
* Batch Normalization: Batch normalization is a method used to make artificial neural networks faster and more stable through normalization of the input layer by adjusting and scaling the activations.

Each of these techniques can help mitigate overfitting, and they are often used together for better performance.

### Using Pretrained Networks (Transfer Learning)
Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It's popular in deep learning because it allows us to train deep neural networks with comparatively little data. In other words, if a model was trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.

#### Using a Pretrained Network in PyTorch
PyTorch provides a number of pretrained networks through the torchvision library. These networks have been trained on the ImageNet dataset which includes over 1 million images from 1000 categories.

Let's consider an example where we want to use the pretrained ResNet-18 model. This model can be loaded in PyTorch using `torchvision.models.resnet18(pretrained=True)`.

Here is a simple code to use a pretrained model in PyTorch:

```python
import torchvision.models as models

# Load the pretrained model
net = models.resnet18(pretrained=True)

# If you want to fine-tune only the top layer of the model, set as below
for param in net.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
net.fc = nn.Linear(net.fc.in_features, 10)  # 100 is an example.

# Forward pass.
outputs = net(images)
# In this example, images would be a tensor containing a batch of images. 
# The images are normalized using the mean and standard deviation of the images in the dataset.
```

This script will first download the pretrained ResNet-18 model when you run it for the first time. It then sets `requires_grad == False` to freeze the parameters so that the gradients are not computed in `backward()`.

Finally, the script replaces the top layer of ResNet which was previously used to classify the images into one of the 1000 categories, with a new layer that classifies images into one of 10 categories (as an example).

In a subsequent training step, only the weights of the final layer will be updated, while the other weights will remain the same. This is because setting `requires_grad == False` for a parameter implies that we do not want to update these weights.

This is the essence of transfer learning. You take an existing model trained on a large dataset, replace the final layer(s) with some layers of your own, and then train the model on your own dataset.

### Advanced CNN Architectures: ResNet, DenseNet, etc.
#### ResNet (Residual Network)
ResNet, introduced in the paper "Deep Residual Learning for Image Recognition" by Kaiming He et al., was the winner of ILSVRC 2015. It introduced a novel architecture with "skip connections" (also known as "shortcuts" or "residual connections").
The main innovation of ResNet is the introduction of the "identity shortcut connection", which skips one or more layers. The ResNet model structure is mainly a stack of Conv2D, BatchNorm, and ReLU layers, and it has an "identity shortcut connection" that skips over the non-linear layers.

The advantage of a residual network is that it can effectively handle the "vanishing gradient problem" which tends to occur when training deep neural networks. By using skip connections, the gradient can be directly backpropagated to shallow layers.

In PyTorch, you can use a pretrained ResNet model in the following way:

```python
import torchvision.models as models

resnet = models.resnet50(pretrained=True)
```

#### DenseNet (Densely Connected Network)
DenseNet, introduced in the paper "Densely Connected Convolutional Networks" by Gao Huang et al., is an extension to ResNet. In DenseNet, each layer is connected to every other layer in a feed-forward fashion.

Whereas traditional convolutional networks with L layers have L connections, one between each layer and its subsequent layer, DenseNet has L(L+1)/2 direct connections. For each layer, the feature maps of all preceding layers are used as inputs, and its own feature maps are used as inputs into all subsequent layers.

Here is an example of how to use a pretrained DenseNet model in PyTorch:

```python
import torchvision.models as models

densenet = models.densenet161(pretrained=True)
```

These models have a significant number of hyperparameters and layers. That's why these are typically used with transfer learning. These models have seen wide use in various tasks beyond just image classification, including object detection and segmentation.

Let's look at another popular deep learning achitecture: recurrence.

## Introduction to RNNs
Recurrent neural networks, or RNNs, are networks with loops in them, allowing information to persist. These are called recurrent because they perform the same task for every element of a sequence, with the output being dependent on the previous computations. Another way to think about RNNs is that they have a 'memory' that captures information about what has been calculated so far.

In theory, RNNs can make use of information in arbitrarily long sequences, but in practice, they are limited to looking back only a few steps due to what’s called the vanishing gradient problem. We'll talk more about this issue shortly.

Here is a simple implementation of a basic RNN in PyTorch:

```python
import torch
from torch import nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=10)
```

In this code, `input_size` is the number of input features per time step, `hidden_size` is the number of features in the hidden state, and `output_size` is the number of output features.

### Issues in Training RNNs
One of the main issues when training RNNs is the so-called vanishing gradient problem. This problem refers to the issue where the gradients of the loss function decay exponentially quickly when propagating back in time, which makes the network hard to train. This happens because the backpropagation of errors involves repeatedly multiplying gradients through every time step, so the gradients can vanish if they have values less than one.

The exploding gradients problem is another issue, which is the opposite of the vanishing gradients problem. Here, the gradient gets too large, resulting in an unstable network. In practice, this can be dealt with by a technique called gradient clipping, which essentially involves scaling down the gradients when they get too large.

A very popular type of RNN that mitigates the vanishing gradient problem are the Long Short-Term Memory (LSTM) unit and Gated Recurrent Unit (GRU).

### Long Short-Term Memory (LSTM)
LSTMs are a special type of RNN specifically designed to avoid the vanishing gradient problem. LSTMs do not really have a fundamentally different architecture from RNNs, but they use a different function to compute the hidden state.

The key to LSTMs is the cell state, which runs straight down the entire chain, with only some minor linear interactions. It’s kind of like a conveyor belt. Information can be easily placed onto the cell state and easily taken off, without any "turbulence" caused by the RNN's complex functions.

This is an oversimplified view of the LSTM, but the main point is that LSTMs have a way to remove or add information to the cell state, carefully regulated by structures called gates.

Here is a basic implementation of an LSTM in PyTorch:

```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)  
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input.view(1, 1, -1), hidden)
        output = self.hidden2out(output.view(1, -1))
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

lstm = SimpleLSTM(input_size=10, hidden_size=20, output_size=10)
```

### Gated Recurrent Unit (GRU)
GRU is a newer generation of recurrent neural networks and is pretty similar to LSTM. But, GRU has two gates (reset and update gates) compared to the three gates (input, forget, and output gates) of LSTM. The fewer gates in a GRU make it a bit more computationally efficient than an LSTM.

Here is a basic implementation of a GRU in PyTorch:

```python
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.hidden2out(output.view(1, -1))
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

gru = SimpleGRU(input_size=10, hidden_size=20, output_size=10)
```

### Implementing LSTMs and GRUs in PyTorch
Let's implement these models in PyTorch and use them on a simple task.

Before we start, remember that LSTMs and GRUs expect input data in a specific format: (`seq_len`, `batch`, `input_size`). Here,
* `seq_len` is the length of the sequence,
* `batch` is the batch size (number of sequences), and
* `input_size` is the number of features in the input.

For this example, let's create an LSTM network that takes sequences of length 10, each with 5 features, and returns a single output value.

#### Implementing LSTM

```python
import torch
from torch import nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
```

This code defines an LSTM module that accepts an input sequence, applies LSTM computations to it, and then passes it through a linear layer to obtain the output.

#### Implementing GRU
Now let's implement a GRU model, which is simpler than LSTM since it has fewer gates.

```python
class GRUNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.gru = nn.GRU(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = torch.zeros(1,1,self.hidden_layer_size)

    def forward(self, input_seq):
        gru_out, self.hidden_cell = self.gru(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(gru_out.view(len(input_seq), -1))
        return predictions[-1]
```

### Sequence to Sequence Models, Language Modelling
Sequence-to-Sequence (Seq2Seq) models are a type of model designed to handle sequence inputs and sequence outputs, where the length of the output sequence may not be equal to the length of the input sequence. They are popularly used in tasks like machine translation, chatbots, and any application that requires generating text output of arbitrary length.

Language modelling, on the other hand, is the task of predicting the next word (or character) in a sequence given the previous words (or characters). Both Seq2Seq models and language models are built using the recurrent networks (RNN, LSTM, GRU) we discussed earlier.

Let's dive in!

#### Sequence to Sequence Model
A basic Seq2Seq model is composed of two main components: an encoder and a decoder. Both are recurrent networks (like LSTMs or GRUs) but they serve different purposes. The encoder takes the input sequence and compresses the information into a context vector (usually the final hidden state of the encoder). The decoder takes this context vector and generates the output sequence.

Here's a simple implementation of a Seq2Seq model using PyTorch:

```python
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.GRU(input_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, output_dim)

    def forward(self, src):
        outputs, hidden = self.encoder(src)
        output, hidden = self.decoder(hidden)
        return output
```

To use the Seq2Seq model, you'll need to provide the model with input sequences. Create some dummy data to pass the to the class:

```python
import torch

# Assume we have 10 sequences, each of length 5, and each element in the sequence has 20 features
input_dim = 20
hidden_dim = 30
output_dim = 10
seq_len = 5
n_seqs = 10

model = Seq2Seq(input_dim, hidden_dim, output_dim)
input_data = torch.randn(seq_len, n_seqs, input_dim)  # Random input data

output = model(input_data)

print(output.shape)  # Should be [seq_len, n_seqs, output_dim]`
```

This model takes an `input_dim` sized input and uses an encoder to create a `hidden_dim` sized context vector. The decoder then takes this context vector and generates an `output_dim` sized output.

#### Language Modelling
In a language model, we're given a sequence of words (or characters), and our task is to predict the next word. In this case, both the input and output sequences have the same length. Here is a simple implementation of a character-level language model using LSTM and PyTorch:

```python
class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        
        out = self.dropout(r_output)
        
        out = out.contiguous().view(-1, self.n_hidden)
        
        out = self.fc(out)
        
        return out, hidden
```

To use the `CharRNN` model for language modeling, you'll need a sequence of characters. Let's create some then see what the model outputs:

```python
import torch

# Define set of all possible characters
tokens = ['a', 'b', 'c', 'd', 'e']

# Convert tokens to integers
token_to_int = {ch: ii for ii, ch in enumerate(tokens)}

# Assume we have a text sequence
text = "abcde"

# Convert characters to integers
input_data = [token_to_int[ch] for ch in text]

# Initialize the model
model = CharRNN(tokens, n_hidden=256, n_layers=2, drop_prob=0.5)

# Convert list of integers to PyTorch tensor
input_data = torch.tensor(input_data)

# Add batch dimension
input_data = input_data.unsqueeze(0)

# Forward pass through the model
output, hidden = model(input_data, None)

print(output.shape)  # Should be [1, len(text), len(tokens)]
```

### Implementing LSTMs and GRUs on real-world data
So far, we've learned the basic theory behind LSTMs and GRUs, as well as how to implement them using PyTorch. Now, we're going to learn how to apply these techniques to real-world data.

One common application for LSTMs and GRUs is in natural language processing (NLP) tasks, where we have sequences of words or characters as input. A common task in NLP is text classification, where the goal is to assign a label to a piece of text. Let's go through an example of using LSTMs for sentiment analysis on movie reviews, using the IMDB dataset available in `torchtext`.

Firstly, you'll need to download and preprocess the data:

```python
from torchtext.legacy import data, datasets

# set up fields
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)

# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)

# build the vocabulary
TEXT.build_vocab(train, vectors='glove.6B.100d')
LABEL.build_vocab(train)

# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_size=64, device=device)
Next, let's define our model. We'll use an LSTM followed by a fully connected layer for classification:
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        return self.fc(hidden)
```

Now we can create an instance of our `RNN` class, and define our training loop:

```python
# create an RNN
model = RNN(len(TEXT.vocab), 100, 256, 1, 2, True, 0.5, TEXT.vocab.stoi[TEXT.pad_token])
model = model.to(device)

# define optimizer and loss
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# define metric
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# training loop
model.train()
for epoch in range(10):
    for batch in train_iter:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Acc: {acc.item()}')
```

This will train the LSTM on the IMDB dataset, and print the training loss and accuracy at each step.

We are only showing the training loop here, but in a real scenario, you would also want to include a validation loop to monitor the model's performance on unseen data, and prevent overfitting. You'd also want to save the model's parameters after training.

More advanced techniques and different applications can be found in the my pytorch-intermediate, pytorch-advanced and pytorch-rnn repos.

### Applications and Limitations of RNNs
Recurrent Neural Networks (RNNs), including their variants like Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRUs), are powerful tools for modeling sequence data. Let's discuss some of their real-world applications, and also their limitations.

#### Real-world Applications of RNNs
* Natural Language Processing (NLP): RNNs are heavily used in NLP tasks because of their effectiveness in handling sequence data. Some applications in NLP include:
    * Text Generation: Given a sequence of words (or characters), predict the next word (or character) in the sequence. This could be used to generate entirely new text that matches the style of the input.
    * Machine Translation: Translate a text from one language to another. A Seq2Seq model is commonly used for this task, where the source language text is input to the encoder, and the decoder generates the text in the target language.
    * Sentiment Analysis: Determine the sentiment expressed in a piece of text. The text is input to the RNN, which then outputs a sentiment value.
    * Speech Recognition: Convert spoken language into written text. The sequence of audio data is input to the RNN, which outputs the corresponding text.
* Time Series Prediction: RNNs can be used to predict future values in a time series, given past values. This is useful in many fields such as finance (e.g., stock prices), weather forecasting, and health (e.g., heart rate data).
* Music Generation: Similar to text generation, RNNs can be used to generate entirely new pieces of music that match the style of a given input sequence.
* Video Processing: RNNs can also be used in video processing tasks like activity recognition, where the goal is to understand what's happening in a video sequence.

#### Limitations of RNNs
While RNNs are powerful, they do come with some limitations:
* Difficulty with long sequences: While LSTMs and GRUs alleviate this issue to some extent, RNNs in general struggle with very long sequences due to the vanishing gradients problem. This is when gradients are backpropagated through many time steps, they can get very small (vanish), which makes the network harder to train.
* Sequential computation: RNNs must process sequences one element at a time, which makes it hard to fully utilize modern hardware that is capable of parallel computations.
* Memory of past information: In theory, RNNs should be able to remember information from the past, in practice, they tend to forget earlier inputs when trained with standard optimization techniques.

There are different ways to address these limitations, including using attention mechanisms (which we'll cover later), or using architectures like the Transformer, which are designed to overcome these issues.

## Generative Adversarial Networks (GANs)
Generative Adversarial Networks, or GANs, are a fascinating idea in deep learning. The goal of GANs is to generate new, synthetic data that resembles some existing real data.

### What is a Generative Adversarial Network (GAN)?
GANs consist of two parts: a generator and a discriminator. The generator creates the data, and the discriminator evaluates the data. The fascinating thing is that these two parts are in a game (hence the name adversarial). The generator tries to generate data that looks real, while the discriminator tries to tell apart the real data from the fake. They both get better over time until, hopefully, the generator generates real-looking data and the discriminator can't tell if it's real or fake.

Let's use a simple analogy to understand this better. Think of a police investigator (the discriminator) and a counterfeiter (the generator). The counterfeiter wants to create counterfeit money that the investigator can't distinguish from real money. In the beginning, the counterfeiter might not be good, and the investigator catches him easily. But as time goes by, the counterfeiter learns from his mistakes and becomes better, and so does the investigator. In the end, the counterfeiter becomes so good at his job that the investigator can't distinguish the counterfeit money from the real ones.

### Basic GAN Implementation
Let's look at a very simplified PyTorch code of how a GAN could be implemented. Note that this is a simplified version and actual implementation could vary based on the type of GAN and the specific task.

```python
import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Here we are defining our generator network
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Here we are defining our discriminator network
        )

    def forward(self, input):
        return self.main(input)

# Creating instances of generator and discriminator
netG = Generator()
netD = Discriminator()

# Establish convention for real and fake labels
real_label = 1
fake_label = 0

# Setup Adam optimizers for G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002)
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002)

# Loss function
criterion = nn.BCELoss()

# Number of epochs
num_epochs = 5

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ## Train with all-real batch
        netD.zero_grad()
        real = data[0]
        batch_size = real.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float)
        output = netD(real).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        ## Train with all-fake batch
        noise = torch.randn(batch_size, 100)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
```

Here we have two neural networks, the `Generator` (G) and the `Discriminator` (D), each with its own optimizer. They play a two-player min-max game with the value function V(G,D):

minG maxD V(D,G) = E[log(D(x))] + E[log(1 - D(G(z)))].

In simple terms, D tries to maximize its probability of assigning the correct label to both training examples and samples from G, and G tries to minimize the probability that D will predict its samples as being fake.

## Variational Autoencoders (VAEs)
Autoencoders are a type of neural network architecture used for learning efficient codings of input data. They are "self-supervised," which means they learn from the input data itself, without the need for labels. They consist of an encoder, which compresses the input data, and a decoder, which reconstructs the original data from the compressed version.

Variational Autoencoders are a special type of autoencoder with added constraints on the encoded representations being learned. More specifically, they are a type of probabilistic approach to encoding where the data is transformed into a normal distribution. When decoding, samples from this distribution are transformed back into the input data. This approach helps generate more robust models and also helps generate new data.

### Basic VAE Implementation
Below is a simplified version of a VAE implemented in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # mu layer
        self.fc22 = nn.Linear(400, 20) # logvariance layer

        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

This VAE is designed to work on MNIST-like data (28x28 grayscale images, thus 784 pixels in total), and it's composed of two main parts: the encoder and the decoder.

The encoder consists of two fully-connected layers that take the input x and output two parameters: `mu` (mean) and `logvar` (logarithm of the variance). These parameters represent the learned normal distribution.

The reparameterization step is a trick that allows us to backpropagate gradients through the random sampling operation. It generates a latent vector `z` by taking a random sample from the normal distribution defined by `mu` and `logvar`.

The decoder then takes the latent vector `z` and reconstructs the input data.

VAEs introduce a probabilistic spin into the world of autoencoders, opening the door for a host of applications and improvements.

## Reinforcement Learning
Reinforcement Learning (RL) is a branch of machine learning that trains software agents to make a sequence of decisions. The agent learns to perform actions by trial and error, by receiving rewards or penalties for these actions.

Let's dive a bit deeper into the terminology of RL:
* Agent: The learner and decision maker.
* Environment: The "world", through which the agent moves.
* Action (A): What the agent can do. For example, moving up, down, left or right.
* State (S): The current scenario returned by the environment.
* Reward (R): An immediate return sent back from the environment to evaluate the last action.
* Policy (π): The strategy that the agent uses to determine the next action based on the current state. It's a map from state to action.
* Value (V or Q): The future reward that an agent would receive by taking an action in a particular state.

There are many ways to implement RL agents, one of the most basic one is using Q-Learning.

### Q-Learning
Q-Learning is a values based algorithm in reinforcement learning. It uses a table (Q-table) where we calculate the maximum expected future rewards for action at each state. The goal is to maximize the value function Q. The Q function is defined as the immediate reward plus the maximum expected future reward.

#### Q-Learning with PyTorch
Let's now discuss how to implement a very basic Q-learning model with PyTorch. Suppose we're training an agent to play a simple game.

```python
import torch
import torch.nn as nn
import numpy as np

# Simple model for Q learning
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        hidden_size = 8
        
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)

state_size = 5
action_size = 2
qnetwork = QNetwork(state_size, action_size)

# Example state
state = np.array([1, 0, 0, 0, 0])

# Convert state to tensor
state_tensor = torch.tensor(state, dtype=torch.float32)

# Compute Q values
q_values = qnetwork(state_tensor)

print("Q values:", q_values.detach().numpy())
```

In this code, we define a simple neural network that takes a state (in our case, a 5-dimensional vector) and outputs a Q value for each possible action (in our case, 2 possible actions). The state represents the current condition of the environment, and the action is what our agent can do.

This is a very simplistic example, and actual implementations will include more features such as an experience replay buffer to store and recall past experiences, an optimizer to update our `QNetwork` weights, a target Q network for more stable learning, and an epsilon-greedy strategy for exploration vs exploitation.
