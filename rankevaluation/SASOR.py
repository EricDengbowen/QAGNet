import pickle as pkl
import numpy as np
import os
import cv2
import pandas as pd
import copy
import pickle
import psutil


def calc_iou(mask_a, mask_b):
    intersection = (mask_a + mask_b >= 2).astype(np.float32).sum()
    iou = intersection / (mask_a + mask_b >= 1).astype(np.float32).sum()
    return iou


def match(matrix, iou_thread, img_name):
    matched_gts = np.arange(matrix.shape[0])
    matched_ranks = matrix.argsort()[:, -1]
    for i, j in zip(matched_gts, matched_ranks):
        if matrix[i][j] < iou_thread:
            matched_ranks[i] = -1
    if len(set(matched_ranks[matched_ranks != -1])) < len(matched_ranks[matched_ranks != -1]):
        for i in set(matched_ranks):
            if i >= 0:
                index_i = np.nonzero(matched_ranks == i)[0]
                if len(index_i) > 1:
                    score_index = matched_ranks[index_i[0]]
                    ious = matrix[:, score_index][index_i]
                    max_index = index_i[ious.argsort()[-1]]
                    rm_index = index_i[np.nonzero(index_i != max_index)[0]]
                    matched_ranks[rm_index] = -1
    if len(set(matched_ranks[matched_ranks != -1])) < len(matched_ranks[matched_ranks != -1]):
        print(img_name)
        raise KeyError
    if len(matched_ranks) < matrix.shape[1]:
        for i in range(matrix.shape[1]):
            if i not in matched_ranks:
                matched_ranks = np.append(matched_ranks, i)
    return matched_ranks


def get_rank_index(gt_masks, segmaps, iou_thread, rank_scores, name):
    segmaps[segmaps > 0.5] = 1
    segmaps[segmaps <= 0.5] = 0

    ious = np.zeros([len(gt_masks), len(segmaps)])
    for i in range(len(gt_masks)):
        for j in range(len(segmaps)):
            ious[i][j] = calc_iou(gt_masks[i], segmaps[j])
    matched_ranks = match(ious, iou_thread, name)
    unmatched_index = np.argwhere(matched_ranks == -1).squeeze(1)
    matched_ranks = matched_ranks[matched_ranks >= 0]
    rank_scores = rank_scores[matched_ranks]
    rank_index = np.array([sorted(rank_scores).index(a) + 1 for a in rank_scores])
    for i in range(len(unmatched_index)):
        rank_index = np.insert(rank_index, unmatched_index[i], 0)
    rank_index = rank_index[:len(gt_masks)]
    return rank_index


def evalu(results, iou_thread):
    print('\nCalculating Sprman ...\n')
    p_sum = 0
    num = len(results)

    for indx, result in enumerate(results):
        print('\r{}/{}'.format(indx+1, len(results)), end="", flush=True)
        #print('memoery_usage:'+str(psutil.virtual_memory().percent)+'%')
        gt_masks = result['gt_masks']
        segmaps = result['segmaps']
        gt_ranks = result['gt_ranks']
        rank_scores = result['rank_scores']
        rank_scores = np.array(rank_scores)[:, None]
        name = result['img_name']

        if len(gt_ranks) == 1:
            num = num - 1
            continue

        gt_index = np.array([sorted(gt_ranks).index(a) + 1 for a in gt_ranks])

        if len(segmaps) == 0:
            rank_index = np.zeros_like(gt_ranks)
        else:
            rank_index = get_rank_index(gt_masks, segmaps, iou_thread, rank_scores, name)

        gt_index = pd.Series(gt_index)
        rank_index = pd.Series(rank_index)
        if rank_index.var() == 0:
            p = 0
        else:
            p = gt_index.corr(rank_index, method='pearson')
        if not np.isnan(p):
            p_sum += p
        else:
            num -= 1

    fianl_p = p_sum/num
    print("SASOR_Score: ")
    print(fianl_p)
    return fianl_p


if __name__ == '__main__':
    f = open('../res.pkl', 'rb')
    results = pickle.load(f)
    evalu(results, 0.5)
