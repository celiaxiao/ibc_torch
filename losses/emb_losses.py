"""EBM loss functions."""

# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
def info_nce(predictions,
             batch_size,
             num_counter_examples,
             softmax_temperature = 1.,
             kl = nn.KLDivLoss(reduction='none')):
  """EBM loss: can you classify the correct example?

  Args:
    predictions: [B x n+1] with true in column [:, -1]
    batch_size: B
    num_counter_examples: n
    softmax_temperature: the temperature of the softmax.
    kl: a KL Divergence loss object

  Returns:
    (loss per each element in the batch, and an optional
     dictionary with any loss objects to log)
  """
  softmaxed_predictions = F.softmax(
      predictions / softmax_temperature, dim=-1)

  # [B x n+1] with 1 in column [:, -1]
  indices = torch.ones((batch_size,), dtype=torch.int64) * num_counter_examples
  labels = F.one_hot(indices, num_classes=num_counter_examples + 1).float()
  # print(softmaxed_predictions, labels)
  # print("labels",labels.shape,"softmax",softmaxed_predictions.shape)
  # torch implementation: loss_pointwise = target * (target.log() - input)
  per_example_loss = kl((softmaxed_predictions+1e-8).log(), labels).sum(dim=-1)
  # print("softmax predition", softmaxed_predictions, "label", labels)
  return per_example_loss, dict()
