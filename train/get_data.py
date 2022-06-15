"""Defines training/eval data iterators."""
#  pylint: disable=g-long-lambda
from data import dataset as x100_dataset_tools
import tensorflow as tf
import tensorflow_datasets as tfds

def to_numpy_dataset(dataset):
    return tfds.as_numpy(dataset)

def get_data_fns(dataset_path,
                 sequence_length,
                 replay_capacity,
                 batch_size,
                 for_rnn,
                 dataset_eval_fraction,
                 flatten_action,
                 norm_function=None,
                 max_data_shards=-1):
  """Gets train and eval datasets."""

  # Helper function for creating train and eval data.
  def create_train_and_eval_fns():
    train_data, eval_data = x100_dataset_tools.create_sequence_datasets(
        dataset_path,
        sequence_length,
        replay_capacity,
        batch_size,
        for_rnn=for_rnn,
        eval_fraction=dataset_eval_fraction,
        max_data_shards=max_data_shards)

    def flatten_and_cast_action(action):
      flat_actions = tf.nest.flatten(action)
      flat_actions = [tf.cast(a, tf.float32) for a in flat_actions]
      return tf.concat(flat_actions, axis=-1)

    if flatten_action:
      train_data = train_data.map(lambda trajectory: trajectory._replace(
          action=flatten_and_cast_action(trajectory.action)))

      if eval_data:
        eval_data = eval_data.map(lambda trajectory: trajectory._replace(
            action=flatten_action(trajectory.action)))

    # We predict 'many-to-one' observations -> action.
    train_data = train_data.map(lambda trajectory: (
        (trajectory.observation, trajectory.action[:, -1, Ellipsis]), ()))
    if eval_data:
      eval_data = eval_data.map(lambda trajectory: (
          (trajectory.observation, trajectory.action[:, -1, Ellipsis]), ()))
      # eval_data = to_numpy_dataset(eval_data)
    # if norm_function:
    #   train_data = to_numpy_dataset(train_data)
    #   train_data = train_data.map(norm_function)
    #   if eval_data:
    #     eval_data = to_numpy_dataset(eval_data)
    #     eval_data = eval_data.map(norm_function)

    return train_data, eval_data

  return create_train_and_eval_fns

if __name__ == '__main__':
    # run 2d particle
    create_train_and_eval_fns = get_data_fns(dataset_path='data/particle/2d_oracle_particle*.tfrecord',
    sequence_length=2, replay_capacity=10000, batch_size=512, for_rnn=False,
    dataset_eval_fraction=0.0, flatten_action=True)

    train_data, test_data = create_train_and_eval_fns()
    print(train_data, test_data)



