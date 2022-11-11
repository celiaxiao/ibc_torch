import os
import sys
from absl import flags

from agents import ibc_agent
from agents.utils import save_config, tile_batch, get_sampling_spec

from network.layers import pointnet
# import more visual layers here if needed

import torch
import pytorch3d
import numpy as np
import tqdm
from torch.utils.data import DataLoader, Dataset

from data.dataset_maniskill import *
from diffuser.datasets.d4rl import get_dataset_from_h5
from setvae.datasets import maniskillDataset
import wandb

from train.visualization import visualize

device = torch.device('cuda')
# General exp info
flags.DEFINE_string('env_name', None, 'Train env name')
flags.DEFINE_string('exp_name', 'experiment', 'the experiment name')
flags.DEFINE_string('control_mode', None, 'Control mode for maniskill envs')
flags.DEFINE_string('obs_mode', None, 'Observation mode for maniskill envs')
flags.DEFINE_string('dataset_dir', None, 'Demo data path')
flags.DEFINE_integer('data_amount', None, 'Number of (obs, act) pair use in training data')

# General training config
flags.DEFINE_integer('batch_size', 512, 'Training batch size')
flags.DEFINE_float('lr', 5e-4, 'Initial optimizer learning rate')
flags.DEFINE_integer('total_steps', int(2e6), 'Total training steps')
flags.DEFINE_integer('step_checkpoint', 1000, 
                     'Save checkpoint every x gradient steps')
flags.DEFINE_integer('viz_freq', 2000, 
                     'Save checkpoint every x gradient steps')
flags.DEFINE_integer('validation_freq', 5000, 
                     'Save checkpoint every x gradient steps')
flags.DEFINE_integer('resume_from_step', None, 
                     'Resume from previous checkpoint')

flags.DEFINE_integer('visual_num_points', None,
                     'Number of points as visual input')
flags.DEFINE_integer('visual_num_channels', 3,
                     '6 if with rgb or 3 only xyz')
flags.DEFINE_integer('visual_output_dim', None,
                     'Dimension for visual network output')
# ShapeNet and maniskill options
flags.DEFINE_string('--maniskill_data_dir', '/home/caiwei/data/soft_body_envs/Fill-v0/Fill-v0_pointcloud_pd_joint_delta_pos_processed.h5', type=str,
                    help='Path to training ShapeNet data')
flags.DEFINE_integer("--split_index", 15000,
                    help='the index where all element before it is used for train set and after it used for validation set')
flags.DEFINE_integer('--dataset_scale', 1.,
                    help='Scale of the dataset (x,y,z * scale = real output, default=1).')
flags.DEFINE_bool('--normalize_per_shape', False,
                    help='Whether to perform normalization per shape.')
flags.DEFINE_bool('--normalize_std_per_axis', False,
                    help='Whether to perform normalization per axis.')
flags.DEFINE_bool('--denormalized_loss', False,
                    help='Whether to perform denormalization before loss computation.')
flags.DEFINE_integer("--tr_max_sample_points", 1024,
                    help='Max number of sampled points (train)')
flags.DEFINE_integer("--te_max_sample_points", 1024,
                    'Max number of sampled points (test)')
flags.DEFINE_bool("--standardize_per_shape", False,
                    'Whether to perform standardization per shape')

FLAGS = flags.FLAGS
FLAGS(sys.argv)
torch.autograd.set_detect_anomaly(True)
def train_autoencoder(config):
    # Create experiment and checkpoint directory
    path = f"work_dirs/formal_exp/{config['env_name']}/{config['exp_name']}/"
    checkpoint_path = path + 'checkpoints/'
    logging_path = path + 'wandb/'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    print(path, checkpoint_path)
    auto_encoder = pointnet.pointNetLayer(in_dim=[config['visual_num_channels'], config['visual_num_points']], out_dim=config['visual_output_dim'], normalize=config['visual_normalize'])
    # prepare training network
    resume_step = config['resume_from_step'] if config['resume_from_step'] else 0
    if resume_step > 0:
        auto_encoder.load_state_dict(torch.load(
        f"{checkpoint_path}step_{resume_step}_pointnet.pt"))
    
    # load dataset
    train_dataset, val_dataset, train_loader, val_loader = maniskillDataset.build(config)

    dataset = get_dataset_from_h5(config['dataset_dir'])
    if config['data_amount']:
        assert config['data_amount'] <= len(dataset), f"Not enough data for {config['data_amount']} pairs!"
        dataset = torch.utils.data.Subset(dataset, range(config['data_amount']))
    

    assert config['obs_dim']==dataset[0][0].size()[0], "obs_dim in dataset mismatch config"
    assert config['act_dim']==dataset[0][1].size()[0], "act_dim in dataset mismatch config"

    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=config['lr'])

    # prepare visualization
    wandb.init(project='ibc', name=config['exp_name'], group=config['env_name'], dir=logging_path, config=config)

    # main training loop
    epoch = 0
    steps = 0
    while steps < config['total_steps']:
        for dictionary in iter(train_loader):
            
            observations = dictionary['set']
            observations = observations.to(device=device, dtype=torch.float)

            reconstruction = auto_encoder(observations)

            chamfer_loss = pytorch3d.loss.chamfer(observations, reconstruction)
            optimizer.zero_grad()
            chamfer_loss.mean().backward()
            optimizer.step()
            steps += 1
            
            wandb.log({
                'loss':chamfer_loss.mean().item(),
            })
            #save checkpoints
            if steps % config['step_checkpoint'] == 0:                
                torch.save(auto_encoder.state_dict(), checkpoint_path+'step_'+steps+resume_step+'_pointnet.pt')
            
            #visualize
            if steps % config['viz_freq'] == 0:
                images = []
                for i in range(min(reconstruction.shape[0], 8)):
                    image = visualize(reconstruction[i])
                    images.append(image)
                wandb.log({"train_reconstruct_image": [wandb.Image(image) for image in images]})
            
            # validation
            if (steps + 1) % config['validation_freq'] == 0:
                validate_autoencoder(val_loader=val_loader, auto_encoder=auto_encoder, path=path, steps=steps)

        
        print("loss at steps", steps, chamfer_loss.mean().item())

def validate_autoencoder(val_loader, auto_encoder, path, steps):
    with torch.no_grad:
        val_data = next(iter(val_loader)) 
        observations = val_data['set']
        observations = observations.to(device=device, dtype=torch.float)

        reconstruction = auto_encoder(observations)

        chamfer_loss = pytorch3d.loss.chamfer(observations, reconstruction)
        
        wandb.log({
            'val_loss':chamfer_loss.mean().item(),
        })
        images = []
        for i in range(min(reconstruction.shape[0], 8)):
            image = visualize(reconstruction[i])
            images.append(image)
        wandb.log({"val_reconstruct_image": [wandb.Image(image) for image in images]})

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    config = FLAGS.flag_values_dict()
    print(config)

    train_autoencoder(config)