

import time
import os
from contextlib import contextmanager
import torch
from tqdm import tqdm
import numpy as np
import copy
import cv2
from rankevaluation.SASOR import evalu as rank_evalu
from rankevaluation.mae_fmeasure_2 import evalu as mf_evalu
import pickle as pkl
from detectron2.data import build_detection_test_loader
from mask2former.data.dataset_mappers.assr_dataset_mapper import AssrDatasetMapper
from mask2former.data.dataset_mappers.irsr_dataset_mapper import IrsrDatasetMapper
from mask2former.data.dataset_mappers.sifr_dataset_mapper import SIFRdataDatasetMapper
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from pycocotools import mask as coco_mask
from rankevaluation.sor_eval import doOffcialSorInference
import detectron2.utils.comm as comm

from mask2former.data.datasets.register_sifr import get_sifr_dicts
from mask2former.data.datasets.register_assr import get_assr_dicts
from mask2former.data.datasets.register_irsr import get_irsr_dicts
from detectron2.data import DatasetFromList, MapDataset

from torch.utils.data import DataLoader,SequentialSampler
import math
from thop import profile
def find_all_indexes(lst, value):
    indexes = []
    for i in range(len(lst)):
        if lst[i] == value:
            indexes.append(i)
    return indexes
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks
@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def inference(cfg, model,model_name,model_root_dir,datasetmode):
    if comm.is_main_process():
        dataset=cfg.EVALUATION.DATASET
        limited=cfg.EVALUATION.LIMITED
        dataPath=cfg.EVALUATION.DATAPATH

        if dataset=="assr":
            SOR_DATASETPATH=dataPath+"ASSR/"
            print('------Evaluation based on ASSR dataset!------')

            assrdataset = get_assr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
            assrdatasetlist = DatasetFromList(assrdataset, copy=False)
            assrdatasetlist = MapDataset(assrdatasetlist, AssrDatasetMapper(cfg, False))
            dataloader = DataLoader(assrdatasetlist, batch_size=1, shuffle=False, num_workers=0,collate_fn=trivial_batch_collator)

        elif dataset=='irsr':
            SOR_DATASETPATH = dataPath+"IRSR/"
            print('------Evaluation based on IRSR dataset!------')

            irsrdataset = get_irsr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
            irsrdatasetlist = DatasetFromList(irsrdataset, copy=False)
            irsrdatasetlist = MapDataset(irsrdatasetlist, IrsrDatasetMapper(cfg, False))
            dataloader = DataLoader(irsrdatasetlist, batch_size=1, shuffle=False, num_workers=0,collate_fn=trivial_batch_collator)

        elif dataset=='sifr':
            SOR_DATASETPATH = dataPath+"SIFR/"
            print('------Evaluation based on SIFR dataset!------')

            sifrdataset = get_sifr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
            sifrdatasetlist = DatasetFromList(sifrdataset, copy=False)
            sifrdatasetlist = MapDataset(sifrdatasetlist, SIFRdataDatasetMapper(cfg, False))
            dataloader = DataLoader(sifrdatasetlist, batch_size=1, shuffle=False, num_workers=0,collate_fn=trivial_batch_collator)

        ourputdir='./evaluationResult/'+model_name.split('.')[0]+'/'
        saliencymapPath = os.path.join(ourputdir,f'GeneratedSaliencyMaps/', model_name+'-ResultThres-'+str(cfg.EVALUATION.RESULT_THRESHOLD)+f'Limited-{limited}'+'/')
        print("------saliencyMapPath is :"+ saliencymapPath)
        if not os.path.exists(saliencymapPath):
            os.makedirs(saliencymapPath)

        with inference_context(model), torch.no_grad():
            res = []
            for idx, inputs in enumerate(dataloader):
                predictions=model(inputs)
                img_height = inputs[0]["height"]
                img_width = inputs[0]["width"]
                if "instances" in predictions[-1]:
                    instances = predictions[-1]["instances"].to("cpu")
                    pred_instances = Instances(instances.image_size)
                    flag = False
                    for index in range(len(instances)):
                        score = instances[index].scores
                        if score > cfg.EVALUATION.RESULT_THRESHOLD:
                            if flag == False:
                                pred_instances = instances[index]
                                flag = True
                            else:
                                pred_instances = Instances.cat([pred_instances, instances[index]])
                    gt_masks_polygon = []
                    gt_ranks = []
                    for ins in inputs[0]['annotations']:
                        gt_ranks.append(ins['gt_rank'])
                        segm = []
                        for seg in ins['segmentation']:
                            segm.append(np.asarray(seg))
                        gt_masks_polygon.append(segm)

                    gt_masks = convert_coco_poly_to_mask(gt_masks_polygon, img_height, img_width).cpu().data.numpy()

                    name = inputs[0]["file_name"].split('/')[-1]

                    if flag:
                        pred_masks = pred_instances.pred_masks.cpu().data.numpy()
                        pred_ranks = pred_instances.pred_rank.cpu().data.numpy()

                        print('len of pred_rank')

                        if limited:
                            if dataset == 'assr':
                                if len(pred_ranks) > 5:
                                    sorted_data = sorted(enumerate(pred_ranks), key=lambda x: x[1], reverse=True)
                                    top5_indices = [index for index, value in sorted_data[:5]]
                                    mask = [i in top5_indices for i in range(len(pred_ranks))]

                                    pred_masks = pred_masks[mask, :, :]
                                    pred_ranks = pred_ranks[mask]
                            elif dataset == 'irsr':
                                if len(pred_ranks) > 8:
                                    sorted_data = sorted(enumerate(pred_ranks), key=lambda x: x[1], reverse=True)
                                    top8_indices = [index for index, value in sorted_data[:8]]
                                    mask = [i in top8_indices for i in range(len(pred_ranks))]

                                    pred_masks = pred_masks[mask, :, :]
                                    pred_ranks = pred_ranks[mask]
                            elif dataset == 'sifr':
                                if len(pred_ranks) > 41:
                                    sorted_data = sorted(enumerate(pred_ranks), key=lambda x: x[1], reverse=True)
                                    top41_indices = [index for index, value in sorted_data[:41]]
                                    mask = [i in top41_indices for i in range(len(pred_ranks))]

                                    pred_masks = pred_masks[mask, :, :]
                                    pred_ranks = pred_ranks[mask]

                        res.append({'gt_masks': [mask for mask in gt_masks], 'segmaps': pred_masks, 'gt_ranks': gt_ranks,'rank_scores': [rank for rank in pred_ranks], 'img_name': name})

                        saliency_rank = [rank for rank in pred_ranks]
                        all_segmaps = np.zeros_like(gt_masks[0], dtype=np.float)
                        segmaps1 = copy.deepcopy(pred_masks)
                        if len(pred_masks) != 0:
                            color_index = [sorted(saliency_rank).index(a) + 1 for a in saliency_rank]
                            color_len = len(color_index)
                            if dataset == 'assr':
                                if color_len <= 10:
                                    color = [math.floor(255. / 10 * (a + (10 - color_len))) for a in color_index]
                                else:
                                    color = [max(math.floor(255. / 10 * (a + (10 - color_len))), 25) for a in color_index]
                            else:
                                color = [255. / len(saliency_rank) * a for a in color_index]
                            cover_region = all_segmaps != 0

                            for i in range(len(segmaps1), 0, -1):
                                obj_id_list = find_all_indexes(color_index, i)
                                if len(obj_id_list) == 0:
                                    continue
                                else:
                                    for obj_id in obj_id_list:
                                        seg = segmaps1[obj_id]
                                        seg[seg >= 0.5] = color[obj_id]
                                        seg[seg < 0.5] = 0
                                        seg[cover_region] = 0
                                        all_segmaps += seg
                                        cover_region = all_segmaps != 0
                            all_segmaps = all_segmaps.astype(np.int)

                            cv2.imwrite(saliencymapPath + '{}.png'.format(name[:-4]), all_segmaps)
                    else:
                        all_segmaps = np.zeros_like(gt_masks[0], dtype=np.int)
                        cv2.imwrite(saliencymapPath + '{}.png'.format(name[:-4]), all_segmaps)
                        segmapsforsasor = np.zeros([0, img_height, img_width])
                        print("Image:"+name+"------Pred_masks is None after confidence threshold")
                        res.append({'gt_masks': [mask for mask in gt_masks], 'segmaps': segmapsforsasor,'gt_ranks': [rank for rank in gt_ranks],'rank_scores': [], 'img_name': name})

            if comm.is_main_process():
                r_corre = rank_evalu(res, 0.5)
                mae, avg_spr_norm, image_used=doOffcialSorInference(SOR_DATASETPATH,saliencymapPath,dataset,datasetmode)
                f_m = mf_evalu(res)
                with open(ourputdir+f"{model_name.split('.')[0]}_{datasetmode}metricsresult_Thres{cfg.EVALUATION.RESULT_THRESHOLD}.txt", "a") as f:
                    f.write('\n------------\n')
                    f.write(f'model_dir: {model_root_dir}\n')
                    f.write(f'model_name: {model_name}\n')
                    f.write("SA-SOR: %.4f\n" % r_corre)
                    f.write('Official SOR: {}\n'.format(avg_spr_norm))
                    f.write('Official Image Used: {}\n'.format(image_used))
                    f.write('MAE: {}\n'.format(mae))
                    f.write('F-measure: {}\n'.format(f_m))
                return mae,avg_spr_norm,image_used,r_corre,f_m

