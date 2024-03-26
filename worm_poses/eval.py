#%%

import os
import itertools
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
#%%








#%%

parser = ArgumentParser()
#ARGUMENTS USED FOR THE LANDMARK PREDICTION
parser.add_argument('--model_path', type=str, default=model_path, help='Path to a pretrained model.')
parser.add_argument('--src_file', type=str, default= src_file, help='Path to the video file to be processed.')
parser.add_argument('--reader_type', type=str, choices=['tierpsy', 'loopbio'], default='loopbio', help='Type of file to be process')
parser.add_argument('--save_dir', type=str, default =save_dir ,help='Path where the results are going to be saved.')
parser.add_argument('--cuda_id', type=int, default=0, help='GPU number to be used')
parser.add_argument('--batch_size', type=int, default=8, help ='Number of images used per training step')
parser.add_argument('--images_queue_size', type=int, default=2, help ='Number of images used per training step')

