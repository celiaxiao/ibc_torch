import torch
def tile_batch(tensors, multiplier):
  '''
  or each tensor t in a (possibly nested structure) of tensors, 
  this function takes a tensor t shaped [batch_size, s0, s1, ...] 
  composed of minibatch entries t[0], ..., t[batch_size - 1] and 
  tiles it to have a shape [batch_size * multiplier, s0, s1, ...] 
  composed of minibatch entries t[0], t[0], ..., t[1], t[1], ... 
  where each minibatch entry is repeated multiplier times.
  '''
  return torch.tile(tensors, (multiplier,)).reshape([-1]+list (tensors.shape[1:]))