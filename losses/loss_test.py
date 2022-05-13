from turtle import shape
import torch
from losses.emb_losses import info_nce
from losses.losses_tf import info_nce_tf
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import numpy as np
def test_emb_loss():
    batch_size = 8
    num_counter_examples = 2
    softmax_temperature=1.0
    preditions_np = np.random.randn(batch_size, num_counter_examples+1)
    kl_tf = tf.keras.losses.KLDivergence(
          reduction=tf.keras.losses.Reduction.NONE)
    kl_torch = nn.KLDivLoss(reduction='none')

    pred_torch = torch.tensor(preditions_np)
    loss_torch, _ = info_nce(pred_torch, batch_size, num_counter_examples, softmax_temperature, kl_torch)

    pred_tf = tf.constant(preditions_np)
    loss_tf,_ = info_nce_tf(pred_tf, batch_size, num_counter_examples, softmax_temperature, kl_tf)
    # print("tf loss", loss_tf.shape)
    # print(softmax_pred_torch, softmax_pred_tf.numpy().astype(dtype=np.float32))
    # assert (softmax_pred_torch.shape== softmax_pred_tf.shape)
    # assert (labels_torch.numpy() == labels_tf.numpy()).all()
    print(loss_torch, loss_tf)
    assert (loss_torch.shape== loss_tf.shape)

if __name__ == "__main__":
    test_emb_loss()
