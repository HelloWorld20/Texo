"""## Part 2: Pruning Demo

### Setup

First, install the required packages and download the datasets and pretrained model. Here we use CIFAR10 dataset and VGG network which is the same as what we used in the Part 1.
"""

print('Installing torchprofile...')
# !pip install torchprofile 1>/dev/null
print('All required packages have been successfully installed!')

import copy
import math
import os
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

from torchprofile import profile_macs

import torch.nn.functional as F
from tutorial_1 import dataloader, demo_images, recover_model, to_image

assert torch.cuda.is_available(), \
"The current runtime does not have CUDA support." \
"Please go to menu bar (Runtime - Change runtime type) and select GPU"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  verbose=True,
) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataloader, desc="eval", leave=False,
                              disable=not verbose):
    # Move the data from CPU to GPU
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Inference
    outputs = model(inputs)

    # Convert logits to class indices
    outputs = outputs.argmax(dim=1)

    # Update metrics
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item()

"""Helper Functions (Flops, Model Size calculation, etc.)"""

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel() # "numel" -> "number of elements"


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

"""Load the MNIST dataset."""

def load_dataset():
    transform=transforms.Compose([
        transforms.ToTensor(), # [0, 255] -> [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)) # mean and std for MNIST
        ])
    to_image = lambda t: (t*0.3081+0.1307).squeeze(0).to('cpu').numpy()

    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = MNIST(
            "data",
            train=(split == "train"),
            download=(split == "train"),
            transform=transform,
        )

    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=256 if split == 'train' else 1000,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True
        )
    
    return to_image, dataloader



"""### Visualize the Demo Images

Create a set of test images for demo.
"""

def create_demo_inputs():
    demos = {0: 3, 1: 2, 2: 1, 3: 30, 4: 4, 5: 15, 6: 11, 7: 0, 8: 61, 9: 9}
    demo_inputs, demo_images = [], []
    for digit, index in demos.items():
        demo_inputs.append(copy.deepcopy(dataset['test'][index][0]))
        demo_images.append(to_image(demo_inputs[-1]))
    demo_inputs = torch.stack(demo_inputs).cuda()
    return demo_images, demo_inputs


def visualize(with_predictions=False):
    plt.figure(figsize=(20, 10))
    predictions = model(demo_inputs).argmax(dim=1) if with_predictions else None
    for digit, index in demos.items():
        plt.subplot(1, 10, digit + 1)
        plt.imshow(demo_images[digit])
        if predictions is None:
            plt.title(f"digit: {digit}")
        else:
            plt.title(f"digit: {digit}\npred: {int(predictions[digit])}")
        plt.axis('off')
    # plt.show()


"""###Model"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # 1 x 32 x 3 x 3 = 288 parameters
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 32 x 64 x 3 x 3=18,432 paramters
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128) # 9216 x 128 = 1,179,648 parameters
        self.fc2 = nn.Linear(128, 10) # 128 x 10 = 1,280 parameters

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# model = Net().cuda()


"""### Pre-train model on MNIST"""

def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: StepLR,
  callbacks = None
) -> None:
  model.train()

  for inputs, targets in tqdm(dataloader, desc='train', leave=False):
    # Move the data from CPU to GPU
    # inputs = inputs.cuda()
    # targets = targets.cuda()

    # Reset the gradients (from the last iteration)
    optimizer.zero_grad()

    # Forward inference
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward propagation
    loss.backward()

    # Update optimizer
    optimizer.step()

    if callbacks is not None:
        for callback in callbacks:
            callback()

  # Update scheduler
  scheduler.step()

def main():
    lr = 1.0
    lr_step_gamma = 0.7
    num_epochs = 1 # 5 epochs for pre-training
    
    # 权重文件路径
    best_weights_path = "/Users/leon.w/workspace/cityu/6009/best_model_weights.pth"
    
    # 检查是否存在最佳权重文件
    if os.path.exists(best_weights_path):
        print(f"=> 找到最佳权重文件 {best_weights_path}，直接加载...")
        best_checkpoint = torch.load(best_weights_path)
        model.load_state_dict(best_checkpoint['state_dict'])
        best_accuracy = best_checkpoint['accuracy']
        print(f"=> 已加载最佳权重，准确率: {best_accuracy:.2f}%")
        recover_model = lambda: model.load_state_dict(best_checkpoint['state_dict'])
        return recover_model

    # 从from torch.optim import *导入。
    # 是一个优化器。咱不用管多大用处
    optimizer = Adadelta(model.parameters(), lr=lr)
    criterion = F.nll_loss # negative log likelihood loss (just cross entropy without log-softmax)
    scheduler = StepLR(optimizer, step_size=1, gamma=lr_step_gamma)

    best_accuracy = 0
    best_checkpoint = dict()
    for epoch in range(num_epochs):
        train(model, dataloader['train'], criterion, optimizer, scheduler)
        accuracy = evaluate(model, dataloader['test'])
        is_best = accuracy > best_accuracy
        if is_best:
            # 如果当前模型的准确率大于之前的最佳准确率，
            # 则将当前模型的状态字典复制到最佳检查点中
            best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_checkpoint['accuracy'] = accuracy
            best_accuracy = accuracy
            
            # 保存最佳权重到文件
            torch.save(best_checkpoint, best_weights_path)
            print(f"=> 保存最佳权重到 {best_weights_path}, 准确率: {best_accuracy:.2f}%")
        print(f'    Epoch {epoch+1:>2d} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')

    print(f"=> loading best checkpoint")
    model.load_state_dict(best_checkpoint['state_dict'])
    recover_model = lambda: model.load_state_dict(best_checkpoint['state_dict'])

    """### Evaluate Dense Model

    Let's first evaluate the accuracy and model size of this model.
    """

    dense_model_accuracy = evaluate(model, dataloader['test'])
    dense_model_size = get_model_size(model)
    print(f"dense model has accuracy={dense_model_accuracy:.2f}%")
    print(f"dense model has size={dense_model_size/MiB:.2f} MiB")
    visualize(with_predictions=True)
    return recover_model



"""### Pruning

Define **pruning functions**.
"""

def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    num_zeros = round(num_elements * sparsity)
    importance = tensor.abs()
    threshold = importance.view(-1).kthvalue(num_zeros).values
    mask = torch.gt(importance, threshold) # "gt" means "greater than"
    tensor.mul_(mask) # element-wise multiplication

    return mask

class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                if isinstance(sparsity_dict, dict): # sparsity_dict is a dictionary
                    masks[name] = fine_grained_prune(param, sparsity_dict[name])
                else: # sparsity_dict can be a list
                    assert(sparsity_dict < 1 and sparsity_dict >= 0)
                    if sparsity_dict > 0:
                        masks[name] = fine_grained_prune(param, sparsity_dict)
        return masks

"""Pruning and evaluate the model"""
def main_pruning():

    sparsity = 0.99 # set a sparsity of 0.99 for all layer types
    recover_model()
    pruner = FineGrainedPruner(model, sparsity)
    pruner.apply(model)

    sparse_model_accuracy = evaluate(model, dataloader['test'])
    sparse_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"{sparsity*100}% sparse model has accuracy={sparse_model_accuracy:.2f}%")
    print(f"{sparsity*100}% sparse model has size={sparse_model_size/MiB:.2f} MiB, "
        f"which is {dense_model_size/sparse_model_size:.2f}X smaller than "
        f"the {dense_model_size/MiB:.2f} MiB dense model")
    visualize(with_predictions=True)

    """### Fine-tunning the pruned model"""

    num_finetune_epochs = 2 # 2 epochs for fine-tunning
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)

    best_sparse_checkpoint = dict()
    best_sparse_accuracy = 0
    print(f'Finetuning Fine-grained Pruned Sparse Model')
    for epoch in range(num_finetune_epochs):
        # At the end of each train iteration, we have to apply the pruning mask
        #    to keep the model sparse during the training
        train(model, dataloader['train'], criterion, optimizer, scheduler,
            callbacks=[lambda: pruner.apply(model)])
        accuracy = evaluate(model, dataloader['test'])
        is_best = accuracy > best_sparse_accuracy
        if is_best:
            best_sparse_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_sparse_accuracy = accuracy
        print(f'    Epoch {epoch+1} Sparse Accuracy {accuracy:.2f}% / Best Sparse Accuracy: {best_sparse_accuracy:.2f}%')

    model.load_state_dict(best_sparse_checkpoint['state_dict'])
    sparse_model_accuracy = evaluate(model, dataloader['test'])
    sparse_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"{sparsity*100}% sparse model has accuracy={sparse_model_accuracy:.2f}%")
    print(f"{sparsity*100}% sparse model has size={sparse_model_size/MiB:.2f} MiB, "
        f"which is {dense_model_size/sparse_model_size:.2f}X smaller than "
        f"the {dense_model_size/MiB:.2f} MiB dense model")
    visualize(with_predictions=True)


to_image, dataloader = load_dataset()
# demo_images, demo_inputs = create_demo_inputs()
# visualize()

recover_model = main()

model = Net()

main_pruning()