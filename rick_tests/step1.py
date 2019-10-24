'''
make annotation set
'''


import sys
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from deepposekit.io import VideoReader, DataGenerator, initialize_dataset
from deepposekit.annotate import KMeansSampler
import tqdm
import glob
import pandas as pd
from os.path import expanduser

## settings
video = r'C:\Users\rick\Desktop\DeepPoseKit_test\test.mp4'
skeleton_file = r'C:\Users\rick\Desktop\DeepPoseKit_test\skeleton.csv'

## sample frames

frames_to_get = 10
reader = VideoReader(video, batch_size=100, gray=True)

randomly_sampled_frames = []
for idx in tqdm.tqdm(range(round(min(frames_to_get/10, len(reader)-1)))):
    batch = reader[idx]
    random_sample = batch[np.random.choice(batch.shape[0], 10, replace=False)]
    randomly_sampled_frames.append(random_sample)
reader.close()

randomly_sampled_frames = np.concatenate(randomly_sampled_frames)
print('frames sampled: %i' % randomly_sampled_frames.shape[0])

## k-means

kmeans = KMeansSampler(n_clusters=4, max_iter=1000, n_init=10, batch_size=100, verbose=True)
kmeans.fit(randomly_sampled_frames)
kmeans_sampled_frames, kmeans_cluster_labels = kmeans.sample_data(randomly_sampled_frames, n_samples_per_label=10)
kmeans_sampled_frames.shape
kmeans.plot_centers(n_rows=2)
plt.show()

## initialize dataset

initialize_dataset(
    images=kmeans_sampled_frames,
    datapath=r'C:\Users\rick\Desktop\DeepPoseKit_test/annotation_set.h5',
    skeleton=skeleton_file,
    overwrite=True # This overwrites the existing datapath
)