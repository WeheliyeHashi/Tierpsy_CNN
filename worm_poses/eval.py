#%%

import os
import itertools
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from inference.track_n_segment import process_file
#%%

src_file = '/home/weheliye@cscdom.csc.mrc.ac.uk/Desktop/deeptangle/Debugging_code/Random_Videos/Video_3/RawVideos/metadata.yaml'
config_path = 24








#%%



parser = ArgumentParser()
#ARGUMENTS USED FOR THE LANDMARK PREDICTION
parser.add_argument('--config_path', type=int, default=config_path, help='Path to the configuration file.')
parser.add_argument('--src_file', type=str, default= src_file, help='Path to the video file to be processed.')
