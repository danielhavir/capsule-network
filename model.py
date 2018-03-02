########################################
#### Licensed under the MIT license ####
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numpy import prod
import capsules as caps

class CapsuleNetwork(nn.Module):
	def __init__(self, img_shape, channels, primary_caps, primary_dim, num_classes, out_dim, num_routing):
		super(CapsuleNetwork, self).__init__()
		self.img_shape = img_shape
		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(img_shape[0], channels, 9, stride=1, bias=True)
		self.relu = nn.ReLU(inplace=True)

		self.primary = caps.PrimaryCapsules(channels, channels, primary_caps, primary_dim)

		self.digits = caps.RoutingCapsules(primary_dim, primary_caps, num_classes, out_dim, num_routing)

		self.decoder = nn.Sequential(
			nn.Linear(out_dim * num_classes, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, int(prod(img_shape)) ),
			nn.Sigmoid()
		)

	def forward(self, x):
		out = self.conv1(x)
		out = self.relu(out)
		out = self.primary(out)
		out = self.digits(out)
		preds = torch.norm(out, dim=-1)

		# Reconstruct the *predicted* image
		_, max_length_idx = preds.max(dim=1)	
		y = Variable(torch.sparse.torch.eye(self.num_classes))
		if torch.cuda.is_available():
			y = y.cuda()

		y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

		reconstructions = self.decoder( (out*y).view(out.size(0), -1) )
		reconstructions = reconstructions.view(-1, *self.img_shape)

		return preds, reconstructions
