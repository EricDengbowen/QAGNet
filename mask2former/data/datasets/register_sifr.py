from detectron2.structures import BoxMode
from pathlib import Path
import json
import cv2

from detectron2.data import MetadataCatalog, DatasetCatalog
import os, json, cv2, random
from detectron2.utils.visualizer import Visualizer

def get_sifr_dicts(root,mode):
    root = Path(root)
    json_file = root / f"{mode}/{mode}_annotations.json"
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, anno in enumerate(imgs_anns):
        record = {}

        filename = str(root / mode/'rgb' / anno['file_name'])
        height, width = cv2.imread(filename).shape[:2]

        img_id=anno['image_id']

        record["file_name"] = filename
        record["image_id"] = img_id
        record["height"] = height
        record["width"] = width

        objs = []
        ranker_order=[]

        for obj_anno in anno['annotations']:
            ranker_order.append(obj_anno['gt_rank'])
            obj={
                "bbox": obj_anno['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": obj_anno['segmentation'],
                "category_id": 0,
                'gt_rank':obj_anno['gt_rank']
            }
            objs.append(obj)

        record['annotations']=objs

        assert len(ranker_order)==len(record['annotations'])
        dataset_dicts.append(record)

    return dataset_dicts[:]


if __name__ == "__main__":
    root='D:/Projects/datasets/SIFR'
    d='test'
    DatasetCatalog.register("sifr_" + d, lambda d=d: get_sifr_dicts(
        root='datasets/sifr', mode=d))
    MetadataCatalog.get("sifr_" + d).set(thing_classes=["salient_obj"])

    sifrdata_metadata=MetadataCatalog.get("sifr_" + d).set(thing_classes=["salient_obj"])
    dataset_dicts = get_sifr_dicts(root,d)
    for a in random.sample(dataset_dicts, 3):
        img = cv2.imread(a["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=sifrdata_metadata, scale=2)
        out = visualizer.draw_dataset_dict(a)
        cv2.imshow('show', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
       #释放窗口
        cv2.destroyAllWindows()