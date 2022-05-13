import numpy as np
import torch
from mcmc import compute_grad_norm
from mcmc_tf import compute_grad_norm_tf
import tensorflow as tf
import os

def test_compute_grad_norm():
    batch_size = 8
    n = 3
    de_dact = np.random.randn(batch_size,n+1)
    d_torch = torch.tensor(de_dact)
    d_tf = tf.constant(de_dact)
    grad_norm_torch = compute_grad_norm(None, d_torch)
    grad_norm_tf = compute_grad_norm_tf(None, d_tf)
    # print(grad_norm_tf, grad_norm_torch)
    assert (grad_norm_tf.numpy() == grad_norm_torch.numpy()).all()

    grad_norm_torch = compute_grad_norm('inf', d_torch)
    grad_norm_tf = compute_grad_norm_tf('inf', d_tf)
    # print(grad_norm_tf, grad_norm_torch)
    assert (np.round(grad_norm_tf.numpy(),4) == np.round(grad_norm_torch.numpy(),4)).all()

    grad_norm_torch = compute_grad_norm('1', d_torch)
    grad_norm_tf = compute_grad_norm_tf('1', d_tf)
    # print(grad_norm_tf, grad_norm_torch)
    assert (np.round(grad_norm_tf.numpy(),4) == np.round(grad_norm_torch.numpy(),4)).all()

    grad_norm_torch = compute_grad_norm('2', d_torch)
    grad_norm_tf = compute_grad_norm_tf('2', d_tf)
    # print(grad_norm_tf, grad_norm_torch)
    assert (np.round(grad_norm_tf.numpy(),4) == np.round(grad_norm_torch.numpy(),4)).all()
    
def test_gradient_wrt_act():
    
    return

if __name__ == "__main__":
    test_compute_grad_norm()