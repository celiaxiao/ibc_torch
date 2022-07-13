import json
import torch
import os.path as osp
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

def dict_flatten(object):
  return torch.concat([torch.flatten(object[key]) for key in object.keys()], axis=-1)

def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if isinstance(obj, dict):
        return {convert_json(k): convert_json(v) 
                for k,v in obj.items()}

    elif isinstance(obj, tuple):
        return (convert_json(x) for x in obj)

    elif isinstance(obj, list):
        return [convert_json(x) for x in obj]

    elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
        return convert_json(obj.__name__)

    elif hasattr(obj,'__dict__') and obj.__dict__:
        obj_dict = {convert_json(k): convert_json(v) 
                    for k,v in obj.__dict__.items()}
        return {str(obj): obj_dict}

    return str(obj)

def save_config(config, dir=''):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        # if self.exp_name is not None:
        #     config_json['exp_name'] = self.exp_name
        # if proc_id() == 0:
        output = json.dumps(config_json, separators=(
            ',', ':\t'), indent=4, sort_keys=True)
        print('Saving config:\n')
        print(output)
        with open(osp.join(dir, "config.json"), 'w') as out:
            out.write(output)

def get_sampling_spec(action_tensor_spec,
                      min_actions,
                      max_actions,
                      uniform_boundary_buffer,
                      ):
    """Defines action sampling based on min/max action +- buffer.

    Args:
        action_tensor_spec: Action spec. dict {'maximum': [], 'minimum': []}
        min_actions: Per-dimension minimum action values seen in subset
        of training data.
        max_actions: Per-dimension minimum action values seen in subset
        of training data.
        uniform_boundary_buffer: Float, percentage of extra "room" to add to
        minimum/maximum boundary when sampling uniform actions.
        act_norm_layer: Normalizer, needed so can sample over normalized actions.
    Returns:
        sampling_spec: Spec used for sampling random uniform negative actions.
    """

    # Optionally add a small buffer of extra acting range.
    action_range = max_actions - min_actions
    min_actions -= action_range * uniform_boundary_buffer
    max_actions += action_range * uniform_boundary_buffer

    # Clip this range to the envs' min/max.
    # There's no point in sampling outside of the envs' min/max.
    min_actions = torch.max(action_tensor_spec['minimum'], min_actions)
    max_actions = torch.min(action_tensor_spec['maximum'], max_actions)

    return min_actions, max_actions
