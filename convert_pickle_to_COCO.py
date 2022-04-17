import pickle
import json
import numpy as np

pkl_path = "/data/syed/mmdet/results/run11_ep4_test_results_2.pkl"

with open(pkl_path, 'rb') as f:
    obj_file = pickle.load(f)

print(type(obj_file))

"""
Notes:
- obj_file: list of length 1000, one for each image
- obj_file[0]: tuple of length 2
- obj_file[0][0]: list of length 3 - bbox object (prediction class and x,y,width,height,confidence)
- obj_file[0][1]: list of length 3 - segm (binary mask for each detected object)
- obj_file[0][0][0]: numpy array of shape (2, 5) - 2 preds for class 0, 5 means x,y,width,height,confidence
- obj_file[0][0][1]: numpy array of shape (3, 5)
- obj_file[0][0][2]: numpy array of shape (6, 5)
- obj_file[0][1][0]: list of length 2 - 2 masks for the 2 preds
- obj_file[0][1][1]: list of length 3
- obj_file[0][1][2]: list of length 6

Sklearn cluster of confidence
Maximum confidence of lower confidence cluster - script in repo
"""
