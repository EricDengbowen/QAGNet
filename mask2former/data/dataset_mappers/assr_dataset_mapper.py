# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
# from . import detection_utils as utils
# from . import transforms as T
from pycocotools import mask as coco_mask
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["AssrDatasetMapper"]
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


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    #assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    # augmentation.extend([
    #     T.ResizeScale(
    #         min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
    #     ),
    #     T.FixedSizeCrop(crop_size=(image_size, image_size)),
    # ])


    augmentation.extend([
        T.Resize((image_size, image_size)),
    ])


    return augmentation

class AssrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """

        # fmt: off
        self.is_train = is_train
        self.image_format = image_format
        self.tfm_gens = tfm_gens
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logging.getLogger(__name__).info(
            "[ASSRdatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)


        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            #dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            gt_ranks = []
            for anno in dataset_dict["annotations"]:
                gt_ranks.append(anno['gt_rank'])
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.

            #instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)


            h, w = instances.image_size

            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances
            dataset_dict["instances"].gt_ranks = torch.LongTensor(gt_ranks)

        return dataset_dict

