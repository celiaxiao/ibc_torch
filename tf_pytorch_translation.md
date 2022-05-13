1. `output = tf.nn.softmax(tensor)`
<--> 
```python
m = torch.nn.Softmax()
input = torch.tensor([-1, 0., 1.])
output = m(input)
```

1. `tf.one_hot(tensor, depth)` <--> `torch.nn.functional.one_hot(tensor, num_classes=depth)`

1. 
```python
kl_tf = tf.keras.losses.KLDivergence(
          reduction=tf.keras.losses.Reduction.NONE)
kl(labels, softmaxed_predictions)
```
<--->
```python
kl_torch = nn.KLDivLoss(reduction='none')
# loss expects the argument preditions in the log-space
kl(softmaxed_predictions.log(), labels).sum(dim=-1)
```

1. `samples = tf.random.categorical(logis, num_samples) `<--> 
`samples=torch.multinomial(torch.exp(logits), num_samples,replacement=True)`

1. `tf.math.bincount(samples, minlength=n, maxlength=n, axis=-1) # samples.shape=(batch_size,n)`
<---> 
```python
temp = torch.arange(batch_size) * n
temp = temp[:,None].expand(input.shape)
torch.bincount((samples+temp).reshape(-1), minlength=n*batch_size).reshape([batch_size,n])
```

1. `tf.nested_utils.tile_batch(tensors, multiplier)`
<-->
`torch.tile(tensors, (multiplier,)).reshape([-1]+list(tensors.shape[1:]))`

1. `tf.identity(tensor)`
<-->
```python
m = nn.Identity()
m(tensor)
```

1. `tf.random.normal(size)`
<--> `torch.normal(0, 1,size=input.shape)`

1. `tf.clip_to_value()` <---> `torch.clamp()`

1. `tf.broadcast_to(input, shape)` <---> `torch.broadcast_to(input, shape)`

