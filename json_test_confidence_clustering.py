import json
import numpy as np
from sklearn.cluster import KMeans

"""
where class_prediction_confs is a dict:
where keys are class names and 
values are lists of all confidence scores generated across the entire set prior to filtering
"""

CLASSES = ["Background", "Glomerulus", "Arteriole", "Artery"]
cutoffs = {}

class_prediction_confs = {
    "Background": [],
    "Glomerulus": [],
    "Arteriole": [],
    "Artery": [],
}

mmdet_segm_test_path = "/data/syed/mmdet/results/run11_ep4_json_results.segm.json"
with open(mmdet_segm_test_path) as f:
    mmdet_annots = json.load(f)

for annot in mmdet_annots:
    class_prediction_confs[CLASSES[annot["category_id"]]].append(annot["score"])


for class_index in CLASSES[1:]:
    class_cutoff_np = np.array(class_prediction_confs[class_index]).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=50, max_iter=500)
    km.fit(class_cutoff_np)
    # get the percents of each size cluster
    low_percent = (len(class_cutoff_np[km.labels_ == np.argmin(km.cluster_centers_)]) / len(class_cutoff_np)) * 100
    high_percent = (len(class_cutoff_np[km.labels_ == np.argmax(km.cluster_centers_)]) / len(class_cutoff_np)) * 100
    # check the size of the clusters; if your smaller cluster set is larger than 70%, increase leniency
    if low_percent > 70:
        print('Large low confidence {} class cluster detected, increasing leniency '
              '({} percent of predictions).'.format(class_index, low_percent))
        cutoffs[class_index] = np.percentile(
            class_cutoff_np[km.labels_ == np.argmin(km.cluster_centers_)], 90)
    # if your larger cluster set is larger than 70%, decrease leniency
    elif high_percent > 70:
        print('Large high confidence {} class cluster detected, decreasing leniency '
              '({} percent of predictions).'.format(class_index, low_percent))
        cutoffs[class_index] = np.percentile(
            class_cutoff_np[km.labels_ == np.argmax(km.cluster_centers_)], 10)
    # otherwise just take the max of the smaller cluster
    else:
        print('{} class clusters are within acceptable range; {} percent low confidence, {} percent high confidence.'.format(class_index, low_percent, high_percent))
        cutoffs[class_index] = np.min(class_cutoff_np[km.labels_ == np.argmax(km.cluster_centers_)])

print(cutoffs)
