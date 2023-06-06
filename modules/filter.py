import torch
from torch import nn

class Filter(nn.Module):

  def __init__(self, n_nodes, memory_dimension,
               device="cpu"):
    super(Filter, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.device = device

    self.__init_filter__()

  def __init_filter__(self):
    """
    Initializes the filter to all zeros. It should be called at the start of each epoch.
    """
    # Treat filter as parameter so that it is saved and loaded together with the model
    self.count = nn.Parameter(torch.zeros((self.n_nodes)).to(self.device),requires_grad=False)
    self.incretment = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.incretment_sqr = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)


  def get_count(self, node_idxs):
    return self.count[node_idxs, :]
  
  def get_incretment(self, node_idxs):
    return self.incretment[node_idxs, :]
  
  def get_incretment_sqr(self, node_idxs):
    return self.incretment_sqr[node_idxs, :]
  
  def detach_filter(self):
    self.incretment.detach_()
  
  def update(self, node_idxs, incret):
    self.incretment[node_idxs, :] = self.incretment[node_idxs, :] + 1
    self.incretment[node_idxs, :] = self.incretment[node_idxs, :] + incret
    self.incretment_sqr[node_idxs, :] = self.incretment_sqr[node_idxs, :] + incret * incret

  def compute_prediction(self, node_idxs):
    mu    = self.incretment[node_idxs, :]/self.count[node_idxs, :]
    sigma = self.incretment_sqr[node_idxs, :]/self.count[node_idxs, :]
    return mu, sigma