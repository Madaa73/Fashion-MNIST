📦 Fashion-MNIST Classification with PyTorch
🧠 Overview

This project implements a complete deep learning pipeline in PyTorch to classify images from the Fashion-MNIST dataset.
It covers dataset preparation, model training, evaluation, and performance tracking using loss and accuracy curves.

The goal is to build a solid understanding of the standard ML workflow:

Dataset → Dataloader → Model → Training Loop → Evaluation

📊 Dataset

We use the Fashion-MNIST dataset, which contains grayscale images of clothing items:

60,000 training samples
10,000 test samples
10 output classes
🏷️ Classes
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
🔧 Data Preprocessing

Images are preprocessed using torchvision transforms:

Resize images to 16×16
Convert to PyTorch tensors
from torchvision import transforms

composed = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor()
])
🏗️ Model Architecture

A simple neural network is used for classification (fully connected or CNN depending on implementation).

Typical flow:

Input: flattened image vector
Hidden layers: linear layers + activation (ReLU)
Output: 10 logits (one per class)
⚙️ Training Configuration
Loss function: CrossEntropyLoss
Optimizer: SGD
Learning rate: 0.1
Epochs: 5
Batch training via DataLoader
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
🔁 Training Pipeline

For each epoch:

Forward pass through the model
Compute loss
Backpropagation
Update weights
Track training loss
Evaluate on validation set
📉 Metrics tracked:
cost_list → average training loss per epoch
accuracy_list → validation accuracy per epoch
📈 Evaluation

Model performance is evaluated using accuracy:

Accuracy = correct predictions / total samples

Evaluation is done in model.eval() mode using torch.no_grad().

📊 Results

Training typically shows:

decreasing loss over epochs
increasing validation accuracy

Example visualization:

plt.plot(cost_list, label="Loss")
plt.plot(accuracy_list, label="Accuracy")
plt.legend()
plt.title("Training Curves")
🚀 How to Run
Install dependencies
pip install torch torchvision matplotlib
Run training
python train.py

or run the notebook:

jupyter notebook

🧠 Key Learnings

This project helped reinforce:

PyTorch dataset & dataloader pipeline
Training loop structure (forward → loss → backward → step)
Model evaluation workflow
Metric tracking across epochs
Debugging ML pipelines in practice
🔮 Future Improvements
Replace MLP with a CNN architecture
Add GPU support (CUDA)
Improve accuracy with better optimization strategies
Add learning rate scheduling
Implement early stopping
Experiment with data augmentation
👤 Madaa73

Built as part of hands-on learning in deep learning and PyTorch fundamentals.
