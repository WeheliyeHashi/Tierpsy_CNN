# %% loading
#%matplotlib qt
from pathlib import Path
import pickle
import jax.numpy as jnp
import jax
import skvideo.io
from deeptangle import build_model, load_model
from deeptangle import checkpoints, logger, utils
import numpy as np
import matplotlib.pyplot as plt
import napari
from deeptangle.predict import non_max_suppression, clean_predictions
from skimage import color
import itertools
import imgstore 
import cv2
import os
import subprocess
import tqdm 
import multiprocessing as mp
import deeptangle as dt
import time
import h5py
import matplotlib.colors as mcolors
from deeptangle.predict import Predictions
import matplotlib.patches as patches
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import (
    FOVMultiWellsSplitter, process_image_from_name)

from tierpsy.helper.params.tracker_param import SplitFOVParams
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader
from tierpsy.features.tierpsy_features import features
import pandas as pd 
import tables as tb
from tierpsy.helper.misc import TimeCounter, print_flush, WLAB, TABLE_FILTERS
from tierpsy.helper.params import set_unit_conversions, read_unit_conversions
from tierpsy.analysis.ske_orient.checkHeadOrientation import isWormHTSwitched
from tierpsy.helper.params.get_defaults import head_tail_defaults
from tierpsy.analysis.compress.Readers import readLoopBio
from deeptangle import spline as sp
from Wormstats import wormstats
from tierpsy.analysis.compress.compressVideo import getROIMask, compressVideo
from tierpsy.helper.params import compress_defaults
from skimage.transform import resize

import json
jax.config.update('jax_platform_name', 'gpu')
print(jax.local_devices())




input_vid = '/home/weheliye@cscdom.csc.mrc.ac.uk/Desktop/deeptangle/Debugging_code/Random_Videos/Video_10_GPCR/RawVideos/20230223/20230223_gpcr_screen_run4_bluelight_20230223_123821.22956818/metadata.yaml'
params_well = 24

#%%
"""
       Initialise the parameters for the model 

"""
def read_params(json_file =''):
         if json_file:
           with open(json_file) as fid:
               params_in_file = json.load(fid)
         return params_in_file


if params_well==24:
    params_in_file = read_params('configs/loopbio_rig_24WP_splitFOV_NN_20220202.json')
elif params_well==96:
     params_in_file = read_params('configs/loopbio_rig_96WP_splitFOV_NN_20220202.json')
else:
     params_in_file = read_params('configs/loopbio_rig_6WP_splitFOV_NN_20220202.json')
     

step_size = params_in_file['step_size']
nframes = params_in_file['nframes']
n_suggestions = params_in_file['n_suggestions']
latent_dim = params_in_file['latent_dim']
skip_frame = params_in_file['skip_frame']
expected_fps = params_in_file['expected_fps']
microns_per_pixel = params_in_file['microns_per_pixel'] 
MW_mapping = params_in_file['MWP_mapping']
model = params_in_file['model_path']
is_light = params_in_file['is_light_background']
block_size = params_in_file['thresh_C']
Constant = params_in_file['thresh_block_size']
max_gap_allowed=max(1, int(expected_fps//2))
window_std =max(int(round(expected_fps)),5)
min_block_size =max(int(round(expected_fps)),5) 
num_batches=10
Save_name = Path(str(Path(input_vid).parent).replace('RawVideos','Results'))
Save_name.mkdir(exist_ok=True, parents=True)
min_frame = 0
max_frame = nframes*11
Start_Frame= 5*skip_frame
#%%%


def _return_masked_image(raw_fname, px2um=microns_per_pixel, json_fname = MW_mapping):
    
    json_fname = Path('Path2JSON').joinpath(MW_mapping)

    splitfov_params = SplitFOVParams(json_file=json_fname)
    shape, edge_frac, sz_mm = splitfov_params.get_common_params()
    uid, rig, ch, mwp_map = splitfov_params.get_params_from_filename(
        raw_fname)
    px2um = px2um

    # read image
    vid = selectVideoReader(str(raw_fname))
    status, img = vid.read_frame(0)

    fovsplitter = FOVMultiWellsSplitter(
        img,
        microns_per_pixel=px2um,
        well_shape=shape,
        well_size_mm=sz_mm,
        well_masked_edge=edge_frac,
        camera_serial=uid,
        rig=rig,
        channel=ch,
        wells_map=mwp_map)
    #fig = fovsplitter.plot_wells()
    return fovsplitter


def  _load_video(raw_videos):
        #store = imgstore.new_for_filename((raw_videos))
        store = selectVideoReader(raw_videos)
        return store

def load_model(origin_dir: str, broadcast: bool = False):
    """
    Builds a model using the weights and the transformation matrix found at the directory.

    Parameters:
        origin_dir: Path to the folder where the weights are.
        broadcast: Whether to broadcast the weights to the number of devices.

    Returns:
        The forward function and the state of the model.
    """
    path = Path(origin_dir)
    with path.joinpath('eigenworms_transform.npy').open('rb') as f:
        A = jnp.load(f)

    forward_fn = build_model(A, n_suggestions, latent_dim, nframes)

    state = checkpoints.restore(origin_dir, broadcast=broadcast)
    return forward_fn, state


def _adpative_thresholding(img : np.uint8, blocksize=block_size, Constant = Constant):
     if is_light:
        img = (255-img)
     th = np.array([cv2.adaptiveThreshold(img[j,:],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blocksize,Constant)==0  for j in (range(img.shape[0])) ])
     return th



def _SVD_bgd(clip):
     X = clip.reshape(clip.shape[0],-1)
     U, s, V = np.linalg.svd(X, full_matrices=False)
     low_rank = np.expand_dims(U[:,0], 1) * s[0] * np.expand_dims(V[0,:], 0)
     X = low_rank.reshape(clip.shape)
     extra_bottom, extra_right = (16-X.shape[0]%16, 16-X.shape[1]%16)
     return X, extra_bottom, extra_right

def non_max_suppression(p, threshold=0.2, overlap_threshold=0.5, cutoff=48):
    p = jax.tree_util.tree_map(lambda x: np.array(x), p)
    non_suppressed_p = dt.non_max_suppression(p, threshold=threshold, overlap_threshold=overlap_threshold, cutoff=cutoff)
    return jax.tree_util.tree_map(lambda x: x[non_suppressed_p], p)

def predict_in_batches(x, forward_fn, state):

            trim_frames = int(len(x) - len(x) % num_batches)
            new_shape = (num_batches, -1, *x[0].shape)
            batched_X = jnp.reshape(x[:trim_frames], new_shape)
            scan_predict_fn = lambda _, u: (None, dt.predict(forward_fn, state, u))
            _, y = jax.lax.scan(scan_predict_fn, init=None, xs=batched_X)
            y = jax.tree_util.tree_map(lambda u: jnp.reshape(u, (-1, *u.shape[2:])), y)
            y = jax.tree_util.tree_map(np.array, y)
            predictions = list(map(dt.Predictions, *y))
            predictions = [non_max_suppression(p) for p in predictions]
                
     
            return predictions



"""
Correct skeleton orientation
"""

def orientWorm(skeleton, prev_skeleton):
    if skeleton.size == 0:
        return skeleton
    if prev_skeleton.size > 0:
        dist2prev_head = np.sum((skeleton - prev_skeleton)**2)
        dist2prev_tail = np.sum((skeleton - prev_skeleton[::-1, :])**2)
        if dist2prev_head > dist2prev_tail:
            # the skeleton is switched
            skeleton = skeleton[::-1, :]
    return skeleton

def correct_skeleton_orient(skeleton):
     skeleton_orientworm = np.zeros(skeleton.shape)
     skeleton_orientworm[0,:] = skeleton[0,:]
     for i in range(1,skeleton.shape[0]):
          skeleton_orientworm[i,:]= orientWorm(skeleton[i,:],skeleton_orientworm[i-1,:])
     return skeleton_orientworm 

"""
Create spline to fill gaps 

"""

def _spline_skel(skeleton, worm_i, step_size=step_size, n_chuncks=10,Ma=5,Mb=10):
    Ma = n_chuncks-4# Number of spline parameters along the time axis (this will have to be tuned) - must be < number of time steps-4
 
    Mb = Mb # Number of spline parameters in each frame (this will have to be tuned) - must be < number of particles-4
    spline_sheet = sp.SplineSheet((Ma,Mb))
    worm_i =(worm_i-worm_i.min())*step_size
    frame_max= worm_i.max()
    #frame_min= worm_i.min()
   
    point_max = skeleton.shape[1]

    samplePoints = np.array([(ell[0], ell[1], t) for el,t in zip(skeleton, worm_i) for ell in (el)])
    sampleParameters = np.array([(t*(Ma-1)/frame_max, s*(Mb-1)/point_max) for t in worm_i for s in range(skeleton.shape[1])])
    
   
    spline_sheet.initializeFromPoints(samplePoints,sampleParameters)

    cparr = [{ "coordinates": c.tolist() } for c in spline_sheet.controlPoints]

    jsondata = {'model': {'controlPoints': cparr,
                    'longitude_count': Ma,
                    'latitude_count': Mb,
            }}
    data= jsondata['model']
    spline = sp.SplineSheet([data['longitude_count'], data['latitude_count']])
    spline.controlPoints  = np.array([x['coordinates'] for x in data['controlPoints']])   
    # with open(f"data/example_spline_{Point_ID_unique}.json", 'w') as fw:
    #     json.dump(jsondata, fw, indent = 4, separators=(',', ': '))
    sampling_rate = frame_max
    num_isolines  = skeleton.shape[1]
    #pts_data = np.flip(spline.controlPoints, 1)
    isolines = []
    for i in range(num_isolines):
        paramlist = [((float(x) * (spline.M[0]-1))/ (sampling_rate-1), (float(i) * (spline.M[1]-1))/ (num_isolines-1)) for x in range(sampling_rate)]
        isolines.append(np.squeeze([spline.parametersToWorld(p) for p in paramlist]))
    
    shape_list = []
    for i in range(len(isolines)):
        shape_list.append(np.flip(isolines[i],1))
    
    shape_list= np.array(shape_list)
    shape_list_array= []
    for i in range(sampling_rate):
        shape_list_array.append(shape_list[:,i,:])

    return np.array(shape_list_array)[:,:,[2,1]]
        
def _begin_process(Save_name, store, min_frame, max_frame, inter_skip, Start_Frame, bgd, extra_bottom, extra_right):
        with h5py.File((Save_name).joinpath('skeletonNN.hdf5'), 'w') as f:
                    
                    w_group = f.create_group('skeletonNN_w')
                    p_group = f.create_group('skeletonNN_p')
                    s_group = f.create_group('skeletonNN_s')
                    batch = []
                    frame_list= []
                    
                    
                    #clip = np.array([store.get_next_image()[0] for _ in range(min_frame, max_frame)])
                    clip = np.array([store.read_frame(frame)[1] for frame in range(min_frame,max_frame)])
                   
                    
                   
                    bn = Path(Save_name).parent.name
                    for time_stamp, frame_number in enumerate(tqdm.trange(0,500,inter_skip, desc = bn)):
                            
                            if is_light:
                                clip_inverted = (255-clip[::10,:])
                            #th = clip_inverted*((clip_inverted- bgd)>15) 
                            th = _adpative_thresholding(clip_inverted)*(clip_inverted)#_adpative_thresholding(255-clip[::10,:])*
                            
                            th = th*((clip_inverted- bgd)>15)
                            th = jnp.pad(th, ((0, 0),(0, extra_bottom), (0, extra_right)),
                                        mode='constant', constant_values=0)  
                            if time_stamp%num_batches==0 and time_stamp!=0:
                                batch = jnp.asarray(batch)
                                predictions = predict_in_batches(batch,forward_fn, state_single)
                                for (Frame, preds) in (zip(frame_list, predictions)):
                                    dataset_name = f'frame_number_{Frame:05}'
                                    #print(Frame, dataset_name)
                                    w,s,p = preds
                                    w_group.create_dataset(dataset_name, data=w, compression='gzip')
                                    p_group.create_dataset(dataset_name, data=p, compression= 'gzip')
                                    s_group.create_dataset(dataset_name, data =s, compression='gzip')
                                batch = []
                                frame_list = []
                            
                            FN = frame_number+(Start_Frame)
                            frame_list.append(int(FN))
                            batch.append((jnp.asarray(th/255,dtype=np.float32)))    
                            small_clip =  np.array([store.get_next_image()[0] for _ in range(inter_skip)])
                            clip = np.concatenate((clip[inter_skip:,:], small_clip), axis=0 )
                        


def _tracking(Results_folder):
     
     """
     Post processing stage 
     """
     with h5py.File(Results_folder, 'r') as f:
        predictions_list = [Predictions(f['skeletonNN_w'][dataset][:],f['skeletonNN_s'][dataset][:],f['skeletonNN_p'][dataset][:])  for dataset in f['skeletonNN_p']]


     identities_list, splines_list = dt.identity_assignment(predictions_list, memory=15)
     identities_list, splines_list = dt.merge_tracks(identities_list, splines_list, framesize=bgd.shape[0])
     return identities_list, splines_list


def _post_processing(Results_folder, splines_list, identities_list, n_chunks=10):

     skeleton_folder = Path(str(Path(input_vid).parent).replace('RawVideos','Results')).joinpath('metadataCNN_featuresN.hdf5')

     wormstats_header= wormstats()
     segment4angle =max(1, int(round(splines_list[0].shape[1]/10)))
     unique_worm_IDS = list( set(x for sublist in identities_list for x in sublist))
     apply_spline = False
     with tb.File(skeleton_folder, 'w') as f:
            tab_time_series = f.create_table('/','timeseries_data', wormstats_header.header_timeseries, filters=TABLE_FILTERS)
            tab_blob = f.create_table('/','blob_features', wormstats_header.blob_data, filters=TABLE_FILTERS)
            tab_traj = f.create_table('/','trajectories_data',wormstats_header.worm_data, filters=TABLE_FILTERS)
            

            worm_coords_array= {}
            for  array_name in ['skeleton']:
                    worm_coords_array[array_name] = f.create_earray(
                        '/',
                        array_name,
                        shape=(
                            0,
                            splines_list[0].shape[1],
                            splines_list[0].shape[2]),
                        atom=tb.Float32Atom(
                            shape=()),
                        filters=TABLE_FILTERS)
            length_tab = 0 
            for i, worm_index_joined in tqdm.tqdm(enumerate(unique_worm_IDS), total = len(unique_worm_IDS)):
                
                worm_i, worm_j = np.where(pd.DataFrame(identities_list).values==worm_index_joined)
                
                

        

                skeleton = np.array([splines_list[x][y,:] for x,y in zip(worm_i,worm_j)])
                skeleton = correct_skeleton_orient(skeleton)
               
                if apply_spline: 
                    skeleton = np.concatenate([_spline_skel(skeleton[i-1:i+n_chunks],worm_i[i-1:i+n_chunks], step_size, n_chunks) for i in range(1,skeleton.shape[0],n_chunks) if len(worm_i[i-1:i+n_chunks])==n_chunks+1])
                    frame_window = np.squeeze([worm_i[i-1:i+n_chunks]*step_size+Start_Frame for i in range(1,skeleton.shape[0],n_chunks) if len(worm_i[i-1:i+n_chunks])==n_chunks+1])
                    frame_window = range(frame_window.min(), frame_window.max())
                else: 
                    frame_window = worm_i*step_size+Start_Frame
                
                skeleton = correct_skeleton_orient(skeleton)
                is_switch_skel, _= isWormHTSwitched(skeleton,segment4angle=segment4angle, max_gap_allowed=max_gap_allowed,window_std=window_std, min_block_size=min_block_size)
                skeleton[is_switch_skel] = skeleton[is_switch_skel,::-1,:]
                
                worm_data=pd.DataFrame([(s, '', worm_index_joined,skel[:,0].mean(), skel[:,1].mean(),84,1,102,100,1)  for s, skel in zip(frame_window, skeleton)], columns=['frame_number','skeleton_id','worm_index_joined','coord_x','coord_y','threshold','has_skeleton','roi_size','area','is_good_skel'])
                worm_data['skeleton_id'] =worm_data.index+length_tab#len(tab_traj)
                timestamp = worm_data['frame_number'].values.astype(np.int32)

                feats = features.get_timeseries_features(skeleton*microns_per_pixel, timestamp = timestamp, fps = expected_fps)
                feats = feats.astype(np.float32)
                feats.insert(0,'worm_index',worm_index_joined)
                feats.insert(2,'well_name', (fovsplitter.find_well_from_trajectories_data(worm_data)))
                blob = feats[['area','length']]
                
                    
                tab_traj.append(worm_data.to_records(index=False))
                set_unit_conversions(tab_traj, 
                                    expected_fps=expected_fps, 
                                microns_per_pixel=microns_per_pixel)
                tab_time_series.append(feats.to_records(index=False))
                tab_blob.append(blob.to_records(index=False))
                worm_coords_array['skeleton'].append(skeleton)
                length_tab = len(tab_traj)
            
                if MW_mapping:
                    fovsplitter= _return_masked_image(input_vid,px2um=microns_per_pixel, json_fname=MW_mapping)
                    fovsplitter.write_fov_wells_to_file(skeleton_folder)

# %%

"""
pre_processing the videos and loading the models  

"""
forward_fn, state = load_model(model, broadcast=False)
state_single =utils.single_from_sharded(state)
store = _load_video(input_vid)

pre_procesing_clip =  np.array([store.read_frame(frame)[1] for frame in range(min_frame, int(store.tot_frames), 500)])
if is_light:
     pre_procesing_clip = 255- pre_procesing_clip
bgd, extra_bottom, extra_right = _SVD_bgd(pre_procesing_clip).min(axis=0)

plt.imshow(bgd)

     

#%%
"""
Execute the code 

"""
store = _load_video(input_vid)
t_s=time.time()
_begin_process(Save_name, store, min_frame, max_frame, step_size,Start_Frame, bgd, extra_bottom, extra_right)
t_end = time.time()   
print(f'total time taken is {(t_end-t_s)/60} minutes')

print('Data processing compeleted')
