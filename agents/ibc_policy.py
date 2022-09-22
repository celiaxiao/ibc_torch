from time import time
from agents import utils, mcmc
import torch
from torchvision.transforms.functional import convert_image_dtype

class MappedCategorical(torch.distributions.Categorical):
    """Categorical distribution that maps classes to specific values."""

    def __init__(self,
               action_spec:int,
               logits=None,
               probs=None,
               mapped_values=None):
        """Initialize Categorical distributions using class log-probabilities.

        Args:
        logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities of a
            set of Categorical distributions. The first `N - 1` dimensions index
            into a batch of independent distributions and the last dimension
            represents a vector of logits for each class. Only one of `logits` or
            `probs` should be passed in.
        probs: An N-D `Tensor`, `N >= 1`, representing the probabilities of a set
            of Categorical distributions. The first `N - 1` dimensions index into a
            batch of independent distributions and the last dimension represents a
            vector of probabilities for each class. Only one of `logits` or `probs`
            should be passed in.
        mapped_values: Values that map to each category.
        """
        self.action_shape = torch.Size([action_spec])
        self._mapped_values = mapped_values
        super(MappedCategorical, self).__init__(
            logits=logits,
            probs=probs)
    
    def sample(self, sample_shape=torch.Size([])):
        """Generate samples of the specified shape."""
        if len(sample_shape) == 0:
          sample_shape = torch.Size([1])
        sample = super(MappedCategorical, self).sample(
            sample_shape=sample_shape)
        sample = sample
        # print('sampling:', sample.shape, self._mapped_values.shape)
        return self._mapped_values[sample]

class IbcPolicy():
  """Class to build Actor Policies."""

  def __init__(self,
               actor_network,
               action_spec:int,
               min_action,
               max_action,
               num_action_samples=2**14,
               training = False,
               use_dfo = False,
               dfo_iterations = 3,
               use_langevin = True,
               langevin_iterations = 100,
               inference_langevin_noise_scale = 1.0,
               optimize_again = False,
               again_stepsize_init = 1e-1,
               again_stepsize_final = 1e-5,
               late_fusion=False,
               obs_norm_layer=None,
               act_denorm_layer=None):
    """Builds an Actor Policy given an actor network.

    Args:
      actor_network: An instance of a `tf_agents.networks.network.Network` to be
        used by the policy. The network will be called with `call(observation,
        step_type, policy_state)` and should return `(actions_or_distributions,
        new_state)`.
      policy_state_spec: A nest of TensorSpec representing the policy_state. If
        not set, defaults to actor_network.state_spec.
      info_spec: A nest of `TensorSpec` representing the policy info.
      num_action_samples: Number of samples to evaluate for every element in the
        call batch.
      clip: Whether to clip actions to spec before returning them.
      training: Whether the network should be called in training mode.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
      use_dfo: Whether to use dfo mcmc at inference time.
      use_langevin: Whether to use Langevin mcmc at inference time.
      inference_langevin_noise_scale: Scale for Langevin noise at inference
        time.
      optimize_again: Whether or not to run another round of Langevin
        steps but this time with no noise.
      again_stepsize_init: If optimize_again, Langevin schedule init.
      again_stepsize_final: If optimize_again, Langevin schedule final.
      late_fusion: If True, observation tiling must be done in the
        actor_network to match the action.
      obs_norm_layer: Use to normalize observations.
      act_denorm_layer: Use to denormalize actions for inference.

    Raises:
      ValueError: if `actor_network` is not of type `network.Network`.
    """
    self._action_spec = action_spec
    self.min_action = torch.nn.Parameter(min_action, requires_grad=False)
    self.max_action = torch.nn.Parameter(max_action, requires_grad=False)
    self._num_action_samples = num_action_samples
    self._use_dfo = use_dfo
    self._dfo_iterations = dfo_iterations
    self._use_langevin = use_langevin
    self._langevin_iterations = langevin_iterations
    self._inference_langevin_noise_scale = inference_langevin_noise_scale
    self._optimize_again = optimize_again
    self._again_stepsize_init = again_stepsize_init
    self._again_stepsize_final = again_stepsize_final
    self._late_fusion = late_fusion
    self._obs_norm_layer = obs_norm_layer
    self._act_denorm_layer = act_denorm_layer

    # TODO(oars): Add validation from the network

    self._actor_network = actor_network
    self._training = training

    super().__init__()

  def act(self, time_step):
    # time_step: dict{'observations'}
    distribution = self._distribution(time_step=time_step)
    sample = distribution.sample([time_step['observations'].shape[0]])
    return sample
    
  def eval(self, time_step):
    action_samples, probs = self._probs(time_step)
    return action_samples[probs.argmax()]

  def _probs(self, time_step):
    observations = time_step['observations']
    if isinstance(observations, dict) and 'rgb' in observations:
      observations['rgb'] = convert_image_dtype(observations['rgb'], dtype=torch.float32)

    if self._obs_norm_layer is not None:
      observations = self._obs_norm_layer(observations)

    if isinstance(observations, dict):
        # print('single obs', observations, list(observations.keys())[0])
        single_obs = observations[list(observations.keys())[0]]
        observations = utils.dict_flatten(observations)
    else:
        single_obs = observations
    batch_size = single_obs.shape[0]
    #TODO: may have error for shape [B, T, obs_dim]
    observations = observations.reshape([batch_size, -1])
    if self._late_fusion:
      maybe_tiled_obs = observations
    else:
      maybe_tiled_obs = utils.tile_batch(observations,
                                              self._num_action_samples)
    # Initialize.
    # TODO(peteflorence): support other initialization options.
    # print('init sample',batch_size, self._num_action_samples, self._action_spec)
    if len(self.min_action) > 1:
      action_samples = \
        torch.distributions.uniform.Uniform(self.min_action,self.max_action).sample(\
            [batch_size, self._num_action_samples])
    else:
      action_samples = \
        torch.distributions.uniform.Uniform(self.min_action,self.max_action).sample(\
            [batch_size, self._num_action_samples, self._action_spec])
    action_samples = action_samples.reshape((batch_size * self._num_action_samples, -1))
    # print('init sample',action_samples.shape, len(self.min_action.shape))
    # MCMC.
    probs = 0
    # print("check random uniform sample shape", action_samples.shape, maybe_tiled_obs.shape)
    if self._use_dfo:
      probs, action_samples, _ = mcmc.iterative_dfo(
          self._actor_network,
          batch_size,
          observations=maybe_tiled_obs,
          action_samples=action_samples,
          policy_state=None,
          num_action_samples=self._num_action_samples,
          min_actions=self.min_action,
          max_actions=self.max_action,
          num_iterations=self._dfo_iterations,
          late_fusion=self._late_fusion,)

    if self._use_langevin:
      action_samples = mcmc.langevin_actions_given_obs(
          self._actor_network,
          observations=maybe_tiled_obs,
          action_samples=action_samples,
          num_action_samples=self._num_action_samples,
          min_actions=self.min_action,
          max_actions=self.max_action,
          num_iterations=self._langevin_iterations,
          noise_scale=1.0).detach()

      # Run a second optimization, a trick for more precise
      # inference.
      if self._optimize_again:
        action_samples = mcmc.langevin_actions_given_obs(
            self._actor_network,
            observations=maybe_tiled_obs,
            action_samples=action_samples,
            num_action_samples=self._num_action_samples,
            min_actions=self.min_action,
            max_actions=self.max_action,
            num_iterations=self._langevin_iterations,
            sampler_stepsize_init=self._again_stepsize_init,
            sampler_stepsize_final=self._again_stepsize_final,
            noise_scale=self._inference_langevin_noise_scale).detach()

      probs = mcmc.get_probabilities(self._actor_network,
                                     batch_size,
                                     self._num_action_samples,
                                     maybe_tiled_obs,
                                     action_samples).detach()

    if self._act_denorm_layer is not None:
      action_samples = self._act_denorm_layer(action_samples)
    return action_samples, probs

  # TODO:time_step.observation must in shape [B, obs_dim], if a single [obs_dim] pass in, will result in wrong batch shape
  def _distribution(self, time_step):
    # Use first observation to figure out batch/time sizes as they should be the
    # same across all observations.
    # action_sample shape [num_policy_sample*batch_size, act_dim]
    # probs shape [num_policy_sample*batch_size]
    action_samples, probs = self._probs(time_step)
    # print("action samples.shape", action_samples.shape, "probs shape", probs.shape)
    # Make a distribution for sampling.
    distribution = MappedCategorical(
        action_spec=self._action_spec, probs=probs, mapped_values=action_samples)
    return distribution

if __name__ == '__main__':
  obs = torch.rand((10,20))
  _distribution()
