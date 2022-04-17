import os
import json

import cv2
import numpy as np
from tqdm import tqdm


def get_contour_area(contour):
    """
    :param contour: Numpy array shape [n_points, 2]
    :return: float of area in pixels of annotation defined by the contour
    """
    c1 = np.expand_dims(contour.astype(np.float32), 1)
    c1 = cv2.UMat(c1)
    return cv2.contourArea(c1)


def polygonFromMask(maskedArr):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:  # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError('No polygons.')
    return segmentation


"""
From clustering script:
Glomerulus class clusters are within acceptable range; 30.070921985815602 percent low confidence, 69.9290780141844 percent high confidence.
Large low confidence Arteriole class cluster detected, increasing leniency (82.53676470588235 percent of predictions).
Artery class clusters are within acceptable range; 60.024829298572314 percent low confidence, 39.975170701427686 percent high confidence.
{'Glomerulus': 0.5822800993919373, 'Arteriole': 0.24685976803302764, 'Artery': 0.5061547160148621}
"""


# Define parts of COCO dataset dictionary
info = {
    "description": "HULA TMA StyleGAN Generated Tiles with Pseudolabels",
    "version": "1.0",
    "year": 2022,
    "contributor": "HULA",
    "date created": "2022/04/15"
}
images = []
annotations = []
categories = [
    {
        "id": 0,
        "name": "Background",
        "supercategory": "Background"
    },
    {
        "id": 1,
        "name": "Glomerulus",
        "supercategory": "Compartment"
    },
    {
        "id": 2,
        "name": "Arteriole",
        "supercategory": "Compartment"
    },
    {
        "id": 3,
        "name": "Artery",
        "supercategory": "Compartment"
    },
]

tile_path = "/data/syed/TMA_4096_generated_crops"
mmdet_segm_test_path = "/data/syed/mmdet/results/run11_ep4_json_results.segm.json"
curr_img_id = 0
curr_annot_id = 0

# Filtering cutoffs obtained from confidence clustering script
FILTERING_CUTOFFS = {'Glomerulus': 0.5822800993919373, 'Arteriole': 0.24685976803302764, 'Artery': 0.5061547160148621}
FILTERING_CUTOFF_LIST = [0, 0.5822800993919373, 0.24685976803302764, 0.5061547160148621]

with open(mmdet_segm_test_path) as f:
    mmdet_annots = json.load(f)

mmdet_annot_filtered = []
for annot in mmdet_annots:
    if annot["score"] > FILTERING_CUTOFF_LIST[annot["category_id"]]:
        mmdet_annot_filtered.append(annot)

all_files = os.listdir(tile_path)
all_files = all_files[0:1000]
print("Total images to process:", len(all_files))

for file in tqdm(all_files):
    image = {
        "id": curr_img_id,
        "width": 4096,
        "height": 4096,
        "file_name": os.path.join(tile_path, file),
    }
    images.append(image)

    # Handle annotations from mmdetection segm prediction json file
    for annot in mmdet_annots:
        if annot["image_id"] == curr_img_id:
            annot["iscrowd"] = 0
            annot["id"] = curr_annot_id
            del annot["score"]
            annotations.append(annot)
            curr_annot_id += 1
        elif annot["image_id"] > curr_img_id:
            break  # Past annotations for this image

    curr_img_id += 1

# Gather all parts of COCO dictionary into one dictionary, dump into json file
coco_dict = {"info": info, "images": images, "annotations": annotations, "categories": categories}

with open('coco_tma_generated_1k_pseudolabeled.json', 'w', encoding='utf-8') as f:
    json.dump(coco_dict, f, ensure_ascii=False, indent=4)

"""
Goal: MMDetection COCO format
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
"""
