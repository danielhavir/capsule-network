########################################
#### Licensed under the MIT license ####
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(s, dim=-1):
	'''
	"Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
	Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
	:param s: 	Vector before activation
	:param dim:	Dimension along which to calculate the norm
	:return: 	Squashed vector
	'''
	squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
	return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)


class PrimaryCapsules(nn.Module):
	def __init__(self, in_channels, out_channels, num_caps, dim_caps,
	kernel_size=9, stride=2, padding=0):
		"""
		Initialize the layer.

		:param in_channels: 	Number of input channels.
		:param out_channels: 	Number of output channels.
		:param num_caps: 		Number of capsules in the capsule layer
		:param dim_caps:		Dimensionality, i.e. length, of the output capsule vector.
		: return:
		"""
		super(PrimaryCapsules, self).__init__()
		self.dim_caps = dim_caps
		self._caps_channel = int(out_channels / 8)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=2, padding=0)

	def forward(self, x):
		out = self.conv(x)
		out = out.view(out.size(0), self._caps_channel, out.size(2), out.size(3), self.dim_caps)
		out = out.view(out.size(0), -1, self.dim_caps)
		return squash(out)


class RoutingCapsules(nn.Module):
	def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing):
		"""
		Initialize the layer.

		:param in_dim: 		Dimensionality (i.e. length) of each capsule vector.
		:param in_caps: 	Number of input capsules if digits layer.
		:param num_caps: 	Number of capsules in the capsule layer
		:param dim_caps: 	Dimensionality, i.e. length, of the output capsule vector.
		:param num_routing:	Number of iterations during routing algorithm
		: return:
		"""
		super(RoutingCapsules, self).__init__()
		self.in_dim = in_dim
		self.in_caps = in_caps
		self.num_caps = num_caps
		self.dim_caps = dim_caps
		self.num_routing = num_routing

		self.W = nn.Parameter( 0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim ) )

	def forward(self, x):
		batch_size = x.size(0)
		# (batch_size, in_caps, in_dim) -> (batch_size, in_caps, 1, in_dim, 1)
		x = x.unsqueeze(1).unsqueeze(4)
		#
		# W @ x =
		# (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
		# (batch_size, num_caps, in_caps, dim_caps, 1)
		u_hat = torch.matmul(self.W, x)
		# (batch_size, num_caps, in_caps, dim_caps)
		u_hat = u_hat.squeeze(-1)
		u_hat_detached = u_hat.detach()

		'''
		Procedure 1: Routing algorithm
		'''
		b = Variable( torch.zeros(batch_size, self.num_caps, self.in_caps, 1) )
		if torch.cuda.is_available():
			b = b.cuda()

		for route_iter in range(self.num_routing-1):
			# (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
			c = F.softmax(b, dim=1)

			# element-wise multiplication
			# (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
			# (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
			# (batch_size, num_caps, dim_caps)
			# s = (c * u_hat).sum(dim=1)
			s = (c * u_hat_detached).sum(dim=2)
			# apply "squashing" non-linearity along dim_caps
			v = squash(s)
			# dot product agreement between the current output vj and the prediction uj|i
			# (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
			# -> (batch_size, num_caps, in_caps, 1)
			uv = torch.matmul(u_hat_detached, v.unsqueeze(-1))
			b += uv
		
		# last iteration is done on the original u_hat, without the routing weights update
		c = F.softmax(b, dim=1)
		s = (c * u_hat).sum(dim=2)
		# apply "squashing" non-linearity along dim_caps
		v = squash(s)

		return v
