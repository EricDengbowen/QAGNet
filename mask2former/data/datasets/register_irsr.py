from detectron2.structures import BoxMode
from pathlib import Path
import json
import cv2
import pickle

from detectron2.data import MetadataCatalog, DatasetCatalog
import os, json, cv2, random
from detectron2.utils.visualizer import Visualizer

def get_irsr_dicts(root, mode):
    root = Path(root)
    pkl_file = root/'Annotations' / f"{mode}.pkl"
    f = open(pkl_file, 'rb')
    imgs_anns=pickle.load(f)
    dataset_dicts = []


    for idx, anno in enumerate(imgs_anns):
        anno_len = len(anno["annotations"])
        if mode == 'train' and anno_len>8:
            continue
        record = {}
        file=anno['file_name'].split('/')[-1]
        filename = str(root / 'Images' / mode / 'rgb'/ file)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for rank, obj_anno in zip(anno['rank'], anno["annotations"]):
            obj = {
                "bbox": obj_anno['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": obj_anno['segmentation'],
                "category_id": 0,
                "gt_rank": rank
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

#      return dataset_dicts[:5]
    return dataset_dicts



if __name__ == "__main__":
    root='D:\Projects\Mask2Former/datasets/IRSR'
    d='train'
    DatasetCatalog.register("irsr" + d, lambda d=d: get_irsr_dicts(
        root='datasets/irsr', mode=d))
    MetadataCatalog.get("irsr" + d).set(thing_classes=["salient_obj"])

    assr_metadata=MetadataCatalog.get("irsr" + d).set(thing_classes=["salient_obj"])
    dataset_dicts = get_irsr_dicts(root,d)
    for a in dataset_dicts:
        # if a["file_name"]=='D:\\Projects\\Mask2Former\\datasets\\IRSR\\Images\\train\\rgb\\COCO_train2014_000000372319.jpg':
            img = cv2.imread(a["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=assr_metadata, scale=2)
            out = visualizer.draw_dataset_dict(a)
            cv2.imshow('show', out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
           #释放窗口
            cv2.destroyAllWindows()
            print(111)
















if __name__ == "__main__":
    root='D:\Projects\Mask2Former/datasets/IRSR'
    d='train'
    DatasetCatalog.register("irsr" + d, lambda d=d: get_irsr_dicts(
        root='datasets/irsr', mode=d))
    MetadataCatalog.get("irsr" + d).set(thing_classes=["salient_obj"])

    assr_metadata=MetadataCatalog.get("irsr" + d).set(thing_classes=["salient_obj"])
    dataset_dicts = get_irsr_dicts(root,d)
    for a in random.sample(dataset_dicts, 3):
        img = cv2.imread(a["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=assr_metadata, scale=2)
        out = visualizer.draw_dataset_dict(a)
        cv2.imshow('show', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
       #释放窗口
        cv2.destroyAllWindows()