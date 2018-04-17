import os
import torch
import torchvision
import torchvision.transforms as transforms
from trainer import CapsNetTrainer
import argparse

DATA_PATH = os.path.join(os.environ['data'])

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# MNIST or CIFAR?
parser.add_argument('dataset', nargs='?', type=str, default='MNIST', help="'MNIST' or 'CIFAR' (case insensitive).")
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
# Select GPU device
parser.add_argument('--gpu_device', type=int, default=None, help='ID of a GPU to use when multiple GPUs are available.')
# Data directory
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the MNIST or CIFAR dataset. Alternatively you can set the path as an environmental variable $data.')
args = parser.parse_args()

if args.gpu_device is not None:
    torch.cuda.set_device(args.gpu_device)

if args.multi_gpu:
    args.batch_size *= torch.cuda.device_count()

datasets = {
    'MNIST': torchvision.datasets.MNIST,
    'CIFAR': torchvision.datasets.CIFAR10
}

if args.dataset.upper() == 'MNIST':
    args.data_path = os.path.join(args.data_path, 'MNIST')
    size = 28
    classes = list(range(10))
    mean, std = ( ( 0.1307,), ( 0.3081,) )
elif args.dataset.upper() == 'CIFAR':
    args.data_path = os.path.join(args.data_path, 'CIFAR')
    size = 32
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    mean, std = ( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
else:
    raise ValueError('Dataset must be either MNIST or CIFAR')

transform = transforms.Compose([
    # shift by 2 pixels in either direction with zero padding.
    transforms.RandomCrop(size, padding=2),
    transforms.ToTensor(),
    transforms.Normalize( mean, std )
])

loaders = {}
trainset = datasets[args.dataset.upper()](root=args.data_path, train=True, download=True, transform=transform)
loaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = datasets[args.dataset.upper()](root=args.data_path, train=False, download=True, transform=transform)
loaders['test'] = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
print(8*'#', f'Using {args.dataset.upper()} dataset', 8*'#')

# Run
caps_net = CapsNetTrainer(loaders, args.batch_size, args.learning_rate, args.num_routing, args.lr_decay, multi_gpu=args.multi_gpu)
caps_net.run(args.epochs, classes=classes)
