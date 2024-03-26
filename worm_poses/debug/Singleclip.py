# %% loading

%matplotlib qt
from pathlib import Path
import pickle
import jax.numpy as jnp
import jax
import skvideo.io
from deeptangle import SyntheticGenerator, build_model, load_model, synthetic_dataset
from deeptangle import checkpoints, logger, utils
import numpy as np
import matplotlib.pyplot as plt
import napari
from deeptangle.predict import non_max_suppression, clean_predictions
from skimage.exposure import equalize_adapthist
from skimage import color
import itertools
import imgstore 
import cv2
from skimage.transform import resize
#%%
#experiment_dir = 'Alans_data/10_frame_skip/models/parameters_multi_worms_adap_thres_skip_10_with_no_background_pca_72_epochs_1000_26_sept_2023_new_ver_comp_2'
#experiment_dir ='Alans_data/25_frame_skip/models/parameters_multi_worms_adap_thres_skip_25_with_no_background_pca_72_epochs_1000_28_sept_2023_new_ver_comp2'#
#experiment_dir = 'Alans_data/10_frame_skip/models/parameters_multi_worms_adap_thres_skip_10_with_no_background_pca_72_epochs_1000_02_oct_2023'
experiment_dir = 'Train_data/Mixed_Frame_clips/models/parameters_multi_worms_Final_mixed_train_data_pca_92_epochs_200_21_Feb_2024_wout_bgd'
input_vid = 'Random_Videos/Video_10_GPCR/RawVideos/20230223/20230223_gpcr_screen_run4_bluelight_20230223_123821.22956818/metadata.yaml'
nframes =11
n_suggestions= 8
latent_dim = 8
min_frame= 3200
max_frame= min_frame+110
skip_frame=10


def  _load_video(raw_videos):
    store = imgstore.new_for_filename((raw_videos))
    return store

store = _load_video(input_vid)

def _adpative_thresholding(img, blocksize=15, Constant =1):
     img = (255-img*255).astype(np.uint8)
     th = np.array([cv2.adaptiveThreshold(img[i,:],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blocksize,Constant)==0  for i in range(11) ])
     return th


def _SVD_bgd(store, max_frame, skip_frame, scale=1):
     clip =255- np.array([store.get_image(store.frame_min+frame)[0] for frame in range(0, max_frame, skip_frame)])
     clip = resize(clip, [clip.shape[0], int(clip.shape[1]/scale), int(clip.shape[2]/scale)])
     X = clip.reshape(clip.shape[0],-1)
     U, s, V = np.linalg.svd(X, full_matrices=False)
     low_rank = np.expand_dims(U[:,0], 1) * s[0] * np.expand_dims(V[0,:], 0)
     X = low_rank.reshape(clip.shape)
     X = resize(X, [X.shape[0], int(X.shape[1]*scale), int(X.shape[2]*scale)])
     return X
#%%
scale = 2
clip = np.array([store.get_image(store.frame_min+frame)[0] for frame in range(min_frame, max_frame, skip_frame)])
bgd = _SVD_bgd(store, store.frame_count, 1000, scale=1)
clip = (255-clip)/255
#th = clip*((clip- bgd.min(axis=0))>0.05) 
#clip =  resize(clip , (clip.shape[0], int(clip.shape[1]/scale),int(clip.shape[2]/scale)))
#bgd =  resize(bgd , (bgd.shape[0], int(bgd.shape[1]/scale),int(bgd.shape[2]/scale)))
clip1= clip[None,:]

th = _adpative_thresholding(clip[:])*clip
th = th*((clip- bgd.min(axis=0))>0.05) 

th[th>0]=1
kernel = np.ones((5,5),np.uint8)
th = cv2.erode(th,kernel,iterations = 1)
#clip  = th
#clip = equalize_adapthist(255-clip)
#clip =clip*0.6

#clip = equalize_adapthist(clip)
#clip =clip*1.5
#X = clip.reshape(clip.shape[0],-1)
#u,s,v = jnp.linalg.svd(X, False)
#mode_range = np.arange(1,5)
#frame_bgd = (np.rint(u[:,mode_range] @ np.diag(s[mode_range]) @ v[mode_range,:])).reshape(clip.shape)
#frame_bgd[frame_bgd<0]=0

#clip=th
clip = jnp.array(th, dtype=np.float32)
viewer =napari.Viewer()
viewer.add_image(th)
#viewer.add_image(frame_bgd)
#viewer.add_image(clip)
#plt.imshow(clip[8,:])

#%%
clip = clip[None, ...]

path = Path(experiment_dir)
with path.joinpath('eigenworms_transform.npy').open('rb') as f:
        A = jnp.load(f)

forward_fn = build_model(A, n_suggestions, latent_dim, nframes)
with path.joinpath('tree.pkl').open('rb') as f:
    tree_struct = pickle.load(f)

leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
with path.joinpath('arrays.npy').open('rb') as f:
    flat_state = [jnp.load(f) for _ in leaves]

state_test = jax.tree_util.tree_unflatten(treedef, flat_state)

#%%
#%matplotlib qt
Batch_number = 3
state_single =utils.single_from_sharded(state_test)
params, state_11, opt_state = state_single
#inputs= X[0,Batch_number,:,:,:]
#Y_label = y [0,Batch_number,:,:,:,:]
inputs = clip#x_label[Batch_number][0,0,:]
#Y_label = y_label[Batch_number][0,0,:]
preds, state_11 = forward_fn.apply(params, state_11, inputs[:], is_training=False)
preds = jax.tree_util.tree_map(lambda x: x[0], preds)
preds = jax.tree_util.tree_map(np.asarray, preds)
best_predictions_idx = non_max_suppression(preds, 0.1, 0.5, 20)
final_predictions = jax.tree_util.tree_map(lambda x: x[best_predictions_idx], preds)
#%%
w,s,p = preds

plt.figure()
plt.xlim(0, clip.shape[3])
plt.ylim(0, clip.shape[2])
plt.imshow((255-clip1[0, 5])/255, cmap="binary")

for x in final_predictions.w[:, 1]:

    plt.plot(x[:, 0], x[:, 1], "-", linewidth=2)
    plt.axis('off')
plt.savefig('Final_splines', dpi=1000)

# %%
p_list = []

for i in range(s.shape[0]):
    if s[i]>1:
        plt.figure(1)
        plt.imshow(clip1[0, 5], cmap="binary")
        plt.scatter(w[i,5,:,0], w[i,5,:,1], linewidths=1)
        #plt.imshow(np.max(resized, axis=2), cmap ='jet', alpha = 0.4)
        #p_list.append(p[i])

        #print(f'The lantent space {p[i]} for step number {i}  and {s[i]}')
plt.axis('off')        
plt.show()
plt.savefig('Final_multiplie_splines', dpi=1000)
# %%
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(np.array(p_list))


# %%
hierarchical_cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(X_pca)
plt.figure(2)
plt.scatter(X_pca[:,0], X_pca[:,1], c =labels, cmap ='jet')
#plt.xlim(-6,8)
#plt.ylim(-2,2)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig('clustermap_for latent_dimesnion', dpi=1000)
# %%
S_conf = np.reshape(s, (44,44,8))

resized = cv2.resize(S_conf, (700,700), interpolation = cv2.INTER_NEAREST)
# %%
plt.figure(3)
#plt.imshow(clip1[0,5].T, cmap="binary")
plt.imshow((np.max(resized, axis=2)), cmap ='jet', alpha = 0.4)
# %%
