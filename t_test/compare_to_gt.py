import json
from itertools import groupby
from sklearn.metrics import jaccard_score

class GT:
    def __init__(self, gt_path):
        self.gt_path = gt_path
        self.gt_data = self.load_gt_file()
        self.gt_data_by_id = self.group_data_by_id(self.gt_data['annotations'])
    

    def load_gt_file(self):
        with open(self.gt_path, 'r') as file:
            data = json.load(file)
        return data
    
    def group_data_by_id(self, data):
        data.sort(key=lambda x: x['id'])
        grouped = {k: list(v) for k, v in groupby(data, key=lambda x: x['id'])}
        return grouped

    def iou_to_gt(self, id, dt_mask):
        gt_mask = self.gt_data_by_id[id]
        return jaccard_score(gt_mask.flatten(), dt_mask.flatten())

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