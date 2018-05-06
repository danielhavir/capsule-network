########################################
#### Licensed under the MIT license ####
########################################

import torch
import torch.nn as nn
import torch.optim as optim
import os
from numpy import prod
from datetime import datetime
from model import CapsuleNetwork
from loss import CapsuleLoss
from time import time

SAVE_MODEL_PATH = 'checkpoints/'
if not os.path.exists(SAVE_MODEL_PATH):
	os.mkdir(SAVE_MODEL_PATH)

class CapsNetTrainer:
	"""
	Wrapper object for handling training and evaluation
	"""
	def __init__(self, loaders, batch_size, learning_rate, num_routing=3, lr_decay=0.9, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), multi_gpu=(torch.cuda.device_count() > 1)):
		self.device = device
		self.multi_gpu = multi_gpu

		self.loaders = loaders
		img_shape = self.loaders['train'].dataset[0][0].numpy().shape
		
		self.net = CapsuleNetwork(img_shape=img_shape, channels=256, primary_dim=8, num_classes=10, out_dim=16, num_routing=num_routing, device=self.device).to(self.device)
		
		if self.multi_gpu:
			self.net = nn.DataParallel(self.net)

		self.criterion = CapsuleLoss(loss_lambda=0.5, recon_loss_scale=5e-4)
		self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
		self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
		print(8*'#', 'PyTorch Model built'.upper(), 8*'#')
		print('Num params:', sum([prod(p.size()) for p in self.net.parameters()]))
	
	def __repr__(self):
		return repr(self.net)

	def run(self, epochs, classes):
		print(8*'#', 'Run started'.upper(), 8*'#')
		eye = torch.eye(len(classes)).to(self.device)
		
		for epoch in range(1, epochs+1):
			for phase in ['train', 'test']:
				print(f'{phase}ing...'.capitalize())
				if phase == 'train':
					self.net.train()
				else:
					self.net.eval()

				t0 = time()
				running_loss = 0.0
				correct = 0; total = 0
				for i, (images, labels) in enumerate(self.loaders[phase]):
					t1 = time()
					images, labels = images.to(self.device), labels.to(self.device)
					# One-hot encode labels
					labels = eye[labels]

					self.optimizer.zero_grad()

					outputs, reconstructions = self.net(images)
					loss = self.criterion(outputs, labels, images, reconstructions)

					if phase == 'train':
						loss.backward()
						self.optimizer.step()

					running_loss += loss.item()

					_, predicted = torch.max(outputs, 1)
					_, labels = torch.max(labels, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum()
					accuracy = float(correct) / float(total)

					if phase == 'train':
						print(f'Epoch {epoch}, Batch {i+1}, Loss {running_loss/(i+1)}',
						f'Accuracy {accuracy} Time {round(time()-t1, 3)}s')
				
				print(f'{phase.upper()} Epoch {epoch}, Loss {running_loss/(i+1)}',
				f'Accuracy {accuracy} Time {round(time()-t0, 3)}s')
			
			self.scheduler.step()
			
		now = str(datetime.now()).replace(" ", "-")
		error_rate = round((1-accuracy)*100, 2)
		torch.save(self.net.state_dict(), os.path.join(SAVE_MODEL_PATH, f'{error_rate}_{now}.pth.tar'))

		class_correct = list(0. for _ in classes)
		class_total = list(0. for _ in classes)
		for images, labels in self.loaders['test']:
			images, labels = images.to(self.device), labels.to(self.device)

			outputs, reconstructions = self.net(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(labels.size(0)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1


		for i in range(len(classes)):
			print('Accuracy of %5s : %2d %%' % (
				classes[i], 100 * class_correct[i] / class_total[i]))
