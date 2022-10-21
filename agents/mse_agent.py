'''
Class of mse agent. 
Updates network according to the mse loss of prediction against observation.
'''
import torch
import torch.nn as nn

class MSEAgent():

    def __init__(self, network, optim):
        super().__init__()
        self.network = network
        self.optim = optim
        self.train_step_counter = 0
        self.loss_fn = nn.MSELoss()

    def train(self, experience):
        predict = self.network(experience[0])
        loss = self.loss_fn(predict, experience[1])
        assert torch.isfinite(loss).all()
        self.optim.zero_grad()
        loss.mean().backward()
        self.optim.step()
        self.train_step_counter += 1
        return {'loss': loss}
