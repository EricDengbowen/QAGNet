import scipy.stats as sc
import os
import pickle
import cv2
from sklearn.metrics import mean_absolute_error
from pycocotools import mask as maskUtils
import numpy as np
import json
import skimage.color
import skimage.io
import skimage.transform
from pathlib import Path
SAL_VAL_THRESH = 0.5



######ASSR dataset
class DatasetTestASSR(object):
    def __init__(self, dataset_root, data_split, eval_spr=None):
        self.dataset_root = dataset_root
        self.data_split = data_split

        self.load_dataset()

        self.eval_spr = eval_spr
        if eval_spr:
            rank_order_root = self.dataset_root + "rank_order/" + self.data_split + "/"
            self.gt_rank_orders = self.load_rank_order_data(rank_order_root)

            obj_seg_data_path = self.dataset_root + \
                                "obj_seg_data_" + self.data_split + ".json"
            self.obj_bboxes, self.obj_seg, self.sal_obj_idx_list, self.not_sal_obj_idx_list = self.load_object_seg_data(obj_seg_data_path)

    def load_dataset(self):
        print("\nLoading Dataset ASSR...")

        image_file = self.data_split + "_images.txt"

        # Get list of image ids
        image_path = os.path.join(self.dataset_root, image_file)
        with open(image_path, "r") as f:
            image_names = [line.strip() for line in f.readlines()]

        self.img_ids = image_names

    def load_rank_order_data(self, rank_order_root):
        # rank_order_data_files = [f for f in os.listdir(rank_order_root)]

        gt_rank_orders = []
        for img_id in self.img_ids:

            p = rank_order_root + img_id + ".json"

            with open(p, "r") as in_file:
                rank_data = json.load(in_file)

            rank_order = rank_data["rank_order"]

            gt_rank_orders.append(rank_order)

        return gt_rank_orders

    def load_object_seg_data(self, obj_data_path):
        with open(obj_data_path, "r") as f:
            data = json.load(f)

        obj_bbox = []
        obj_seg = []
        for i in range(len(data)):
            img_data = data[i]

            img_obj_data = img_data["object_data"]

            _img_obj_bbox = []
            _img_obj_seg = []
            for obj_data in img_obj_data:
                _img_obj_bbox.append(obj_data["bbox"])
                _img_obj_seg.append(obj_data["segmentation"])

            obj_bbox.append(_img_obj_bbox)
            obj_seg.append(_img_obj_seg)

        # Find N salient objects based on gt rank order
        _sal_obj_idx_list = []
        _not_sal_obj_idx_list = []
        # Create a set for defined salient objects
        for i in range(len(obj_bbox)):
            gt_ranks = np.array(self.gt_rank_orders[i])
            _idx_sal = np.where(gt_ranks > SAL_VAL_THRESH)[0].tolist()
            _sal_obj_idx_list.append(_idx_sal)

            _idx_not_sal = np.where(gt_ranks <= SAL_VAL_THRESH)[0].tolist()
            _not_sal_obj_idx_list.append(_idx_not_sal)

        return obj_bbox, obj_seg, _sal_obj_idx_list, _not_sal_obj_idx_list

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """

        # Load image
        p = self.dataset_root + "images/" + self.data_split + "/" + image_id + ".jpg"
        image = skimage.io.imread(p)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_gt_mask(self, image_id):
        # Load mask
        p = self.dataset_root + "gt/" + self.data_split + "/" + image_id + ".png"
        og_gt_mask = cv2.imread(p, 1).astype(np.float32)

        # Need only one channel
        mask = og_gt_mask[:, :, 0]

        # Normalize to 0-1
        mask /= 255.0

        return np.array(mask)

######IRSR dataset
class DatasetTestIRSR(object):
    def __init__(self, dataset_root, data_split, eval_spr=None):
        self.dataset_root = dataset_root
        self.data_split = data_split

        pkl_file = dataset_root+'Annotations/' + f"{self.data_split}.pkl"
        f = open(pkl_file, 'rb')
        self.irsrdata = pickle.load(f)
        print("\nLoading Dataset IRSR...")
        image_names=[]
        rank_orders=[]
        obj_bbox = []
        obj_seg = []
        _sal_obj_idx_list = []
        _not_sal_obj_idx_list = []
        for img_data in self.irsrdata:
            _img_obj_bbox = []
            _img_obj_seg = []
            image_names.append(img_data['file_name'].split('/')[-1][:-4])
            rank_orders.append(img_data['rank'])
            img_obj_data = img_data["annotations"]
            for obj_data in img_obj_data:
                _img_obj_bbox.append(obj_data["bbox"])
                _img_obj_seg.append(obj_data["segmentation"])
            obj_bbox.append(_img_obj_bbox)
            obj_seg.append(_img_obj_seg)

        self.img_ids=image_names
        self.gt_rank_orders=rank_orders

        for i in range(len(obj_bbox)):
            gt_ranks = np.array(self.gt_rank_orders[i])
            _idx_sal = np.where(gt_ranks >=0)[0].tolist()
            _sal_obj_idx_list.append(_idx_sal)

            _idx_not_sal = np.where(gt_ranks < 0)[0].tolist()
            _not_sal_obj_idx_list.append(_idx_not_sal)

        self.obj_bboxes=obj_bbox
        self.obj_seg=obj_seg
        self.sal_obj_idx_list=_sal_obj_idx_list
        self.not_sal_obj_idx_list=_not_sal_obj_idx_list
        self.eval_spr = eval_spr

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """

        # Load image
        p = self.dataset_root + "Images/" + self.data_split + "/rgb/" + image_id + ".jpg"
        image = skimage.io.imread(p)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_gt_mask(self, image_id):
        # Load mask
        p = self.dataset_root + "Images/" + self.data_split + "/gt/" + image_id + ".png"
        og_gt_mask = cv2.imread(p, 1).astype(np.float32)

        # Need only one channel
        mask = og_gt_mask[:, :, 0]

        # Normalize to 0-1
        mask /= 255.0

        return np.array(mask)
######SIFR dataset
class DatasetTestSIFR(object):
    def __init__(self, dataset_root, data_split, eval_spr=None):
        self.dataset_root = dataset_root
        self.data_split = data_split

        json_file=dataset_root+self.data_split+f'/{self.data_split}_annotations.json'
        with open(json_file,'rb') as f:
            self.sifrdata = json.load(f)
        print("\nLoading Dataset SIFR...")

        image_names=[]
        rank_orders=[]
        obj_bbox = []
        obj_seg = []
        _sal_obj_idx_list = []
        _not_sal_obj_idx_list = []
        myheight=[]
        mywidth=[]

        for img_data in self.sifrdata:
            _img_obj_bbox = []
            _img_obj_seg = []
            rank=[]
            image_names.append(img_data['file_name'].split('.')[0])
            img_obj_data = img_data["annotations"]
            myheight.append(img_data['height'])
            mywidth.append(img_data['width'])
            for obj_data in img_obj_data:
                _img_obj_bbox.append(obj_data["bbox"])
                _img_obj_seg.append(obj_data["segmentation"])
                rank.append(obj_data['gt_rank'])
            rank_orders.append(rank)
            obj_bbox.append(_img_obj_bbox)
            obj_seg.append(_img_obj_seg)

        self.img_ids=image_names
        self.gt_rank_orders=rank_orders

        for i in range(len(obj_bbox)):
            gt_ranks = np.array(self.gt_rank_orders[i])
            _idx_sal = np.where(gt_ranks >=0)[0].tolist()
            _sal_obj_idx_list.append(_idx_sal)

            _idx_not_sal = np.where(gt_ranks < 0)[0].tolist()
            _not_sal_obj_idx_list.append(_idx_not_sal)

        self.obj_bboxes=obj_bbox
        self.obj_seg=obj_seg
        self.sal_obj_idx_list=_sal_obj_idx_list
        self.not_sal_obj_idx_list=_not_sal_obj_idx_list
        self.eval_spr = eval_spr
        self.height=myheight
        self.width=mywidth

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """

        # Load image
        p = self.dataset_root + self.data_split + "/rgb/" + image_id + ".jpg"
        image = skimage.io.imread(p)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_gt_mask(self, image_id):
        # Load mask
        p = self.dataset_root +  self.data_split + "/gt/" + image_id + ".png"
        og_gt_mask = cv2.imread(p, 1).astype(np.float32)

        # Need only one channel
        mask = og_gt_mask[:, :, 0]

        # Normalize to 0-1
        mask /= 255.0

        return np.array(mask)

def get_obj_mask(seg_ann_data, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    if isinstance(seg_ann_data, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(seg_ann_data, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(seg_ann_data['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(seg_ann_data, height, width)
    else:
        # rle
        # rle = seg_ann_data['segmentation']
        rle = seg_ann_data

    m = maskUtils.decode(rle)

    return m


# Keep only salient objects that are considered salient by both maps
def get_usable_salient_objects_agreed(image_1_list, image_2_list):
    #gt_ranks, pred_ranks
    # Remove indices list
    rm_list = []
    for idx in range(len(image_1_list)):
        v = image_1_list[idx]
        v2 = image_2_list[idx]

        # if v == 0 or v2 == 0:
        #     rm_list.append(idx)

        if v2 == 0:
            rm_list.append(idx)

    # Use indices list
    use_list = list(range(0, len(image_1_list)))
    use_list = list(np.delete(np.array(use_list), rm_list))

    # Remove the indices
    x = np.array(image_1_list)
    y = np.array(image_2_list)
    x = list(np.delete(x, rm_list))
    y = list(np.delete(y, rm_list))

    return x, y, use_list

WIDTH = 640
HEIGHT = 480

# Percentage of object pixels having predicted saliency value to consider as salient object
# for cases where object segments overlap each other
SEG_THRESHOLD = .5


def load_saliency_map(path):
    # Load mask
    sal_map = cv2.imread(path, 1).astype(np.float32)

    # Need only one channel
    sal_map = sal_map[:, :, 0]

    # Normalize to 0-1
    sal_map /= 255.0

    return sal_map


def eval_mae(dataset, map_path):
    print("Calculating MAE...")

    mae_list = []

    num = len(dataset.img_ids)
    for i in range(num):
        image_id = dataset.img_ids[i]

        p = map_path + image_id + ".png"

        pred_mask = load_saliency_map(p)

        gt_mask = dataset.load_gt_mask(image_id)

        # Flatten masks
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()

        mae = mean_absolute_error(gt_mask, pred_mask)

        mae_list.append(mae)

    print("\n")
    avg_mae = sum(mae_list) / len(mae_list)
    print("Average MAE Images = ", avg_mae)
    return avg_mae


def eval_mae_binary_mask(dataset, map_path):
    print("Calculating MAE (Binary Saliency)...")

    mae_list = []

    num = len(dataset.img_ids)
    for i in range(num):
        image_id = dataset.img_ids[i]

        p = map_path + image_id + ".png"
        pred_mask = load_saliency_map(p)

        gt_mask = dataset.load_gt_mask(image_id)

        # Convert masks to binary
        pred_mask[pred_mask > 0] = 1
        gt_mask[gt_mask > 0] = 1

        # Flatten masks
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()

        mae = mean_absolute_error(gt_mask, pred_mask)

        mae_list.append(mae)

    print("\n")
    avg_mae = sum(mae_list) / len(mae_list)
    print("Average MAE Images (Binary Masks) = ", avg_mae)


def calculate_spr(dataset, model_pred_data_path,datasetname):
    print("Calculating SOR...")

    # Load GT Rank
    gt_rank_order = dataset.gt_rank_orders

    spr_data = []

    num = len(dataset.img_ids)
    for i in range(num):
        # Image Id
        image_id = dataset.img_ids[i]

        sal_obj_idx = dataset.sal_obj_idx_list[i]

        N = len(sal_obj_idx)

        # load seg data
        obj_seg = dataset.obj_seg[i]  # polygon
        instance_masks = []
        instance_pix_count = []

        # Create mask for each salient object
        for s_i in range(len(sal_obj_idx)):
            sal_idx = sal_obj_idx[s_i]

            # Get corresponding segmentation data
            seg = obj_seg[sal_idx]

            # Binary mask of object segment
            if datasetname=='sifr':
                mask = get_obj_mask(seg, dataset.height[i], dataset.width[i])
            else:
                mask = get_obj_mask(seg, HEIGHT, WIDTH)

            # Count number of pixels of object segment
            pix_count = mask.sum()

            instance_masks.append(mask)
            instance_pix_count.append(pix_count)

        # ********** Load Predicted Rank
        pred_data_path = model_pred_data_path + dataset.img_ids[i] + ".png"

        pred_sal_map = cv2.imread(pred_data_path)[:, :, 0]

        # Get corresponding predicted rank for each gt salient objects
        pred_ranks = []

        # Create mask for each salient object
        for s_i in range(len(instance_masks)):
            gt_seg_mask = instance_masks[s_i]
            gt_pix_count = instance_pix_count[s_i]

            pred_seg = np.where(gt_seg_mask == 1, pred_sal_map, 0)

            # number of pixels with predicted values
            pred_pix_loc = np.where(pred_seg > 0)

            pred_pix_num = len(pred_pix_loc[0])

            # Get rank of object
            r = 0
            if pred_pix_num > int(gt_pix_count * SEG_THRESHOLD):

                vals = pred_seg[pred_pix_loc[0], pred_pix_loc[1]]

                mode = sc.mode(vals)[0][0]
                r = mode

            pred_ranks.append(r)


        # ********** Load GT Rank
        gt_rank_order_list = gt_rank_order[i]

        # Get Gt Rank Order of salient objects
        gt_ranks = []
        for j in range(N):
            s_idx = sal_obj_idx[j]
            gt_r = gt_rank_order_list[s_idx]
            gt_ranks.append(gt_r)

        # Remove objects with no saliency value in both list
        gt_ranks, pred_ranks, use_indices_list = \
            get_usable_salient_objects_agreed(gt_ranks, pred_ranks)
        ####pred_ranks 0 is the background black pixels
        ####NOTE!! IRSR and our dataset have the gt_rank 0, ASSR does not have this

        spr = None

        if len(gt_ranks) > 1:
            spr = sc.spearmanr(gt_ranks, pred_ranks)
        elif len(gt_ranks) == 1:
            spr = 1

        d = [image_id, spr, use_indices_list]
        spr_data.append(d)

    # out_root = "../spr_data/"
    # out_path = out_root + "spr_data"
    # if not os.path.exists(out_root):
    #     os.makedirs(out_root)

    # with open(out_path, "wb") as f:
    #     pickle.dump(spr_data, f)
    return spr_data


def extract_spr_value(data_list):
    use_idx_list = []
    spr = []
    for i in range(len(data_list)):
        s = data_list[i][1]

        if s == 1:
            spr.append(s)
            use_idx_list.append(i)
        elif s and not np.isnan(s[0]):
            spr.append(s[0])
            use_idx_list.append(i)
        else:
            # N = 0, not obj ranker_score > 0.5
            pass

    return spr, use_idx_list


def cal_avg_spr(data_list):
    spr = np.array(data_list)
    avg = np.average(spr)
    return avg


def get_norm_spr(spr_value):
    #       m - r_min
    # m -> ---------------- x (t_max - t_min) + t_min
    #       r_max - r_min
    #
    # m = measure value
    # r_min = min range of measurement
    # r_max = max range of measurement
    # t_min = min range of desired scale
    # t_max = max range of desired scale

    r_min = -1
    r_max = 1

    norm_spr = (spr_value - r_min) / (r_max - r_min)

    return norm_spr


def eval_spr(spr_all_data):
    # with open(spr_data_path, "rb") as f:
    #     spr_all_data = pickle.load(f)

    spr_data, spr_use_idx = extract_spr_value(spr_all_data)

    pos_l = []
    neg_l = []
    for i in range(len(spr_data)):
        if spr_data[i] > 0:
            pos_l.append(spr_data[i])
        else:
            neg_l.append(spr_data[i])

    print("Positive SPR: ", len(pos_l))
    print("Negative SPR: ", len(neg_l))

    avg_spr = cal_avg_spr(spr_data)
    avg_spr_norm = get_norm_spr(avg_spr)

    print("\n----------------------------------------------------------")
    print(len(spr_data), "/", len(spr_all_data), " - ",
          (len(spr_all_data) - len(spr_data)), "Images Not used")
    print("Average SPR Saliency: ", avg_spr)
    print("Average SPR Saliency Normalized: ", avg_spr_norm)
    return len(spr_data), avg_spr_norm

def eval_sor(map_path, dataset_root='datasets/ASSR/'):
    data_split = "test"
    dataset = DatasetTestASSR(dataset_root,
                          data_split, eval_spr=True)

    # Calculate MAE
    avg_mae = eval_mae(dataset, map_path)

    ####################################################
    # Calculate SOR
    spr_data = calculate_spr(dataset, map_path)
    image_used, sor = eval_spr(spr_data)
    return {
        'mae': avg_mae,
        'image_used': image_used,
        'sor': sor
    }

def doOffcialSor(data_root, generatedMapPath,datasetname):
    print("Evaluate-do official SOR")
    # DATASET_ROOT = "../datasets/ASSR/"   # Change to your location
    # data_split = "test"
    # dataset = DatasetTestASSR(DATASET_ROOT,data_split, eval_spr=True)

    DATASET_ROOT = data_root  # Change to your location
    data_split = "test"
    if datasetname=='irsr':
        dataset = DatasetTestIRSR(DATASET_ROOT, data_split, eval_spr=True)
    elif datasetname=='assr':
        dataset = DatasetTestASSR(DATASET_ROOT, data_split, eval_spr=True)
    elif datasetname=='sifr':
        dataset=DatasetTestSIFR(DATASET_ROOT, data_split, eval_spr=True)

    ####################################################

    map_path = generatedMapPath

    # Calculate MAE
    mae=eval_mae(dataset, map_path)
    eval_mae_binary_mask(dataset, map_path)

    ####################################################
    # Calculate SOR
    spr_data = calculate_spr(dataset, map_path,datasetname)
    image_used,avg_spr_norm=eval_spr(spr_data)

    return mae, avg_spr_norm, image_used

def doOffcialSorInference(data_root, generatedMapPath,datasetname,datasetmode):
    print("Evaluate-do official SOR")
    DATASET_ROOT = data_root  # Change to your location
    data_split = datasetmode
    if datasetname=='irsr':
        dataset = DatasetTestIRSR(DATASET_ROOT, data_split, eval_spr=True)
    elif datasetname=='assr':
        dataset = DatasetTestASSR(DATASET_ROOT, data_split, eval_spr=True)
    elif datasetname=='sifr':
        dataset=DatasetTestSIFR(DATASET_ROOT, data_split, eval_spr=True)



    ####################################################

    map_path = generatedMapPath

    # Calculate MAE
    mae=eval_mae(dataset, map_path)
    eval_mae_binary_mask(dataset, map_path)

    ####################################################
    # Calculate SOR
    spr_data = calculate_spr(dataset, map_path,datasetname)
    image_used,avg_spr_norm=eval_spr(spr_data)

    return mae, avg_spr_norm, image_used


if __name__ == '__main__':
    print("Evaluate")

    # DATASET_ROOT = "../datasets/ASSR/"   # Change to your location
    # data_split = "test"
    # dataset = DatasetTestASSR(DATASET_ROOT,data_split, eval_spr=True)

    DATASET_ROOT = "../datasets/IRSR/"   # Change to your location
    data_split = "test"
    datasetname="IRSR"
    dataset = DatasetTestIRSR(DATASET_ROOT,data_split, eval_spr=True)

    ####################################################
    map_path = "../generatedmaps/saliencymapIRSR-score0.7segthres05/"

    # Calculate MAE
    eval_mae(dataset, map_path)
    eval_mae_binary_mask(dataset, map_path)

    ####################################################
    # Calculate SOR
    spr_data = calculate_spr(dataset, map_path,datasetname)
    eval_spr(spr_data)
