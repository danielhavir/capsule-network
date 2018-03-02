import os
import torch
import torchvision
import torchvision.transforms as transforms
from trainer import CapsNetTrainer
import argparse

DATA_PATH = os.path.join(os.environ['data'], 'MNIST')

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# Batch size
parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size.')
# Epochs
parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs.')
# Learning rate
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate.')
# Number of routing iterations
parser.add_argument('--num_routing', type=int, default=3, help='Number of routing iteration in routing capsules.')
# Exponential learning rate decay
parser.add_argument('--lr_decay', type=float, default=0.96, help='Exponential learning rate decay.')
# Use multiple GPUs?
parser.add_argument('--multi_gpu', action='store_true', help='Flag whether to use multiple GPUs.')
# Data directory
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Flag whether to use multiple GPUs.')
args = parser.parse_args()

if args.multi_gpu:
    args.batch_size *= torch.cuda.device_count()

transform = transforms.Compose([
    # shift by 2 pixels in either direction with zero padding.
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize( ( 0.1307,), ( 0.3081,) )
])

loaders = {}
trainset = torchvision.datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
loaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
loaders['test'] = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Run
caps_net = CapsNetTrainer(loaders, args.batch_size, args.learning_rate, args.num_routing, args.lr_decay, multi_gpu=args.multi_gpu)
caps_net.run(args.epochs, classes=list(range(10)))
