from detectron2.structures import BoxMode
from pathlib import Path
import json
import cv2

from detectron2.data import MetadataCatalog, DatasetCatalog
import os, json, cv2, random
from detectron2.utils.visualizer import Visualizer
SAL_THR = 0.5

def get_assr_dicts(root, mode):
    root = Path(root)
    json_file = root / f"obj_seg_data_{mode}.json"
    list_file = root / f"{mode}_images.txt"
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, anno in enumerate(imgs_anns):
        record = {}

        filename = str(root / 'images' / mode / (anno['img'] + '.jpg'))
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        with open(root / 'rank_order' / mode / (anno['img'] + '.json')) as f:
            ranker_order = json.load(f)['rank_order']

        objs = []
        assert len(ranker_order) == len(
            anno["object_data"]), "Every box should correspond a rank order"

        filtered_list = [value for value in ranker_order if value > 0.5]
        reallength=len(filtered_list)
        for rank, obj_anno in zip(ranker_order, anno["object_data"]):
            # 这里要过滤一下rank <= 0.5 的 box
            if rank > SAL_THR:
                obj = {
                    "bbox": obj_anno['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": obj_anno['segmentation'],
                    "category_id": 0,
                    "gt_rank": int(rank * 10 - 6-(5-reallength))  # map 0.5~1.0 to 0,1,2,3,4,also consider the number of objects less than 5
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == "__main__":
    root='D:\Projects\Mask2Former/datasets/ASSR'
    d='test'
    DatasetCatalog.register("assr_" + d, lambda d=d: get_assr_dicts(
        root='datasets/ASSR', mode=d))
    MetadataCatalog.get("assr_" + d).set(thing_classes=["salient_obj"])

    assr_metadata=MetadataCatalog.get("assr_" + d).set(thing_classes=["salient_obj"])
    dataset_dicts = get_assr_dicts(root,d)
    for a in random.sample(dataset_dicts, 3):
        img = cv2.imread(a["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=assr_metadata, scale=2)
        out = visualizer.draw_dataset_dict(a)
        cv2.imshow('show', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
       #释放窗口
        cv2.destroyAllWindows()