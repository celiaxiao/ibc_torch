"""Defines various constants shared across ibc."""
IMG_KEYS = ['rgb', 'front', 'image']

"""Defines all the tasks supported. Used to define enums in train_eval, etc."""

IBC_TASKS = ['REACH', 'PUSH', 'INSERT', 'PARTICLE', 'PUSH_DISCONTINUOUS',
             'PUSH_MULTIMODAL']
ADROIT_TASKS = ['pen-human-v0', 'hammer-human-v0', 'door-human-v0',
                'relocate-human-v0',]
D4RL_TASKS = ['antmaze-large-diverse-v0',
              'antmaze-large-play-v0',
              'antmaze-medium-diverse-v0',
              'antmaze-medium-play-v0',
              'halfcheetah-expert-v0',
              'halfcheetah-medium-expert-v0',
              'halfcheetah-medium-replay-v0',
              'halfcheetah-medium-v0',
              'hopper-expert-v0',
              'hopper-medium-expert-v0',
              'hopper-medium-replay-v0',
              'hopper-medium-v0',
              'kitchen-complete-v0',
              'kitchen-mixed-v0',
              'kitchen-partial-v0',
              'walker2d-expert-v0',
              'walker2d-medium-expert-v0',
              'walker2d-medium-replay-v0',
              'walker2d-medium-v0'] + ADROIT_TASKS
