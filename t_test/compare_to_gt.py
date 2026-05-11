import json
from itertools import groupby
from pycocotools import mask as mask_utils
import numpy as np
from collections import defaultdict

def mask_to_rle(gt_mask, height, width):
    if isinstance(gt_mask, list):
        # Even if it's one object, COCO polygons are wrapped in a list
        rles = mask_utils.frPyObjects(gt_mask, height, width)
        return mask_utils.merge(rles)

    # 2. Check if it's an RLE (Dictionary)
    elif isinstance(gt_mask, dict):
        if isinstance(gt_mask['counts'], list):
            # It's an uncompressed RLE (integer list), needs encoding
            return mask_utils.frPyObjects(gt_mask, height, width)
        else:
            # It's already a compressed RLE (bytes/string)
            return gt_mask
        
def mask_to_binary(mask, height, width):
    """
    Converts any COCO segmentation format into a 2D binary numpy array.
    """
    # 0. Check if it's already a numpy array (binary mask)
    if isinstance(mask, np.ndarray):
        return mask

    # 1. Check if it's a Polygon (list of lists of floats)
    if isinstance(mask, list):
        rles = mask_utils.frPyObjects(mask, height, width)
        compressed_rle = mask_utils.merge(rles)
        return mask_utils.decode(compressed_rle)

    # 2. Check if it's an RLE (Dictionary)
    elif isinstance(mask, dict):
        if isinstance(mask['counts'], list):
            # Uncompressed RLE (integer list)
            compressed_rle = mask_utils.frPyObjects(mask, height, width)
            return mask_utils.decode(compressed_rle)
        else:
            # Compressed RLE (bytes/string)
            return mask_utils.decode(mask)
            
    else:
        raise ValueError(f"Unknown mask format: {type(mask)}")
    
class GT:
    def __init__(self, gt_path):
        self.gt_path = gt_path
        self.gt_data = self.load_gt_file()
        self.gt_data_by_id = self.group_data_by_id(self.gt_data['annotations'], self.gt_data['images'])
    

    def load_gt_file(self):
        with open(self.gt_path, 'r') as file:
            data = json.load(file)
        return data
    
    def group_data_by_id(self, data, img_data):
        img_lookup = {img['id']: (img['height'], img['width']) for img in img_data}  
        grouped = defaultdict(list)
        for anno in data:
            grouped[anno['id']] = anno
        final_grouped = {}
        for id, annotations in grouped.items():
            img_id = annotations['image_id']
            if img_id in img_lookup:
                h, w = img_lookup[img_id]
                # We return a dict containing the list of masks + the dimensions
                final_grouped[id] = {
                    'annotations': annotations,
                    'height': h,
                    'width': w
                }
        return final_grouped

    def iou_to_gt(self, id, dt_mask):
        gt_mask = self.gt_data_by_id[id]['annotations']['segmentation']
        height = self.gt_data_by_id[id]['height']
        width = self.gt_data_by_id[id]['width']
            
        dt_rle = mask_utils.encode(np.asfortranarray(dt_mask.astype(np.uint8)))
        gt_rle = mask_to_rle(gt_mask, height, width)
        return mask_utils.iou([gt_rle], [dt_rle], [0])

    def eval_set(self, eval_arr):
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval

        cocoGt = COCO(self.gt_path)
        cocoDt = cocoGt.loadRes(eval_arr)

        cocoEval = COCOeval(cocoGt, cocoDt, 'segm', sigmas=None, use_area=True)
        # if save_data_dir[:4] == "COCO":
        #     cocoEval.params.areaRng[0] = [1024, 1e10]

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()