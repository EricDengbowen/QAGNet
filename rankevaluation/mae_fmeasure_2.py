import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error


def calu_mae(gt_ranks, rank_scores, gt_masks, segmaps, names):
    print('\nCalculating MAE\n')
    num = len(gt_ranks)
    mae = 0
    for i in range(len(gt_ranks)):
        gt_map, rank_map = make_map(gt_ranks[i], rank_scores[i], gt_masks[i], segmaps[i])
        gt_map = gt_map.flatten()
        rank_map = rank_map.flatten()
        mae += mean_absolute_error(gt_map, rank_map)
        print('\r{}/{}'.format(i+1, len(gt_ranks)), end="", flush=True)
    return mae / num


def calu_fmeasure(gt, pred):
    betaSqr = 0.3
    positive_set = gt
    P = np.sum(positive_set)
    positive_samples = pred
    TPmat = positive_set * positive_samples
    PS = np.sum(positive_samples)
    TP = np.sum(TPmat)
    TPR = TP / P
    Precision = TP/PS
    if PS == 0:
        F = 0
        Precision = 0
        TPR = 0
    elif TPR == 0:
        F = 0
    else:
        F = (1 + betaSqr) * TPR * Precision / (TPR + betaSqr * Precision)
    return F


def make_map(gt_rank, rank_score, gt_masks, segmaps):
    image_shape = segmaps.shape[1:]
    gt_map = np.zeros(image_shape)
    rank_map = np.zeros(image_shape)
    gt_index = (np.asarray(gt_rank) + 1).astype(np.float)/len(gt_rank)
    rank_index = [sorted(rank_score).index(a) for a in rank_score]
    rank_index = (np.asarray(rank_index) + 1).astype(np.float)/len(rank_index)
    for i in range(len(segmaps)):
        rank_map[segmaps[i] >= 0.5] = rank_index[i]
    for i in range(len(gt_masks)):
        gt_map[gt_masks[i] != 0] = gt_index[i]
    return gt_map, rank_map


def f_measure(gt_ranks, rank_scores, gt_masks, segmaps):
    print('\nCalculating F-measure ...\n')
    num = len(gt_ranks)
    f_final = 0
    for i in range(len(gt_ranks)):
        gt_map, rank_map = make_map(gt_ranks[i], rank_scores[i], gt_masks[i], segmaps[i])
        instances = np.unique(gt_map)
        f = 0
        for j in range(len(instances)-1):
            thr_low, thr_high = instances[j], instances[j+1]
            gt_mask = (gt_map > thr_low) & (gt_map <= thr_high)
            rank_mask = (rank_map > thr_low) & (rank_map <= thr_high)
            f_level_j = calu_fmeasure(gt_mask, rank_mask)
            f += f_level_j
        try:
            f = f / (len(instances)-1)
        except:
            num -= 1
            continue
        f_final += f
        print('\r{}/{}'.format(i+1, len(gt_ranks)), end="", flush=True)
    return f_final / num


def evalu(results):
    gt_ranks = [r.pop("gt_ranks") for r in results]
    rank_scores = [r.pop("rank_scores") for r in results]
    gt_masks = [r.pop("gt_masks") for r in results]
    segmaps = [r.pop("segmaps") for r in results]
    names = [r.pop("img_name") for r in results]
    f_m = f_measure(gt_ranks, rank_scores, gt_masks, segmaps)
    #mae = calu_mae(gt_ranks, rank_scores, gt_masks, segmaps, names)
    return f_m


if __name__ == '__main__':
    f = open('../res.pkl', 'rb')
    res = pickle.load(f)
    results = evalu(res)
    print(results)
    pass
