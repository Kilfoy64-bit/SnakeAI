try:
	import sys
	import torch
	import torch.nn as nn
	import torch.optim as optim
	import torch.nn.functional as F
	import os
except ImportError as err:
    print (f"couldn't load module. {err}")
    sys.exit(2)

class Linear_QNet(nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super().__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = self.linear2(x)
		return x

	def save(self, file_name='model.pth'):
		model_folder_path = './model'
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)
		
		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)

class QTrainer:
	def __init__(self, policy_net, target_net, lr, gamma, device):
		self.lr = lr
		self.gamma = gamma
		# self.model = model

		self.policy_net = policy_net
		self.target_net = target_net
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()

		self.optimizer = optim.RMSprop(self.policy_net.parameters())
		self.criterion = nn.SmoothL1Loss()
	
	def train_step(self, state, action, reward, next_state, game_over):
		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.float)
		reward = torch.tensor(reward, dtype=torch.float)

		if len(state.shape) == 1:
			# (1, x)
			state = torch.unsqueeze(state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			action = torch.unsqueeze(action, 0)
			reward = torch.unsqueeze(reward, 0)
			game_over = (game_over, )
		
		# 1: preditcted Q values with current state
		prediction = self.policy_net(state)
		target = prediction.clone()

		# Q_new = reward + gamma * max(next_predicted Q value)
		for i in range(len(game_over)):
			Q_new = reward[i]
			if not game_over[i]:
				Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
			
			target[i][torch.argmax(action).item()] = Q_new
		
		self.optimizer.zero_grad()
		loss = self.criterion(target, prediction)
		loss.backward()
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1,1)

		self.optimizer.step()

class DQN(nn.Module):

	def __init__(self, h, w, outputs):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32
		self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		# x = x.to(self.device)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

	def save(self, file_name='model.pth'):
		model_folder_path = './model'
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)
		
		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)