"""
Detectron2 training script with a plain training loop.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import sys
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import logging
from inference import inference
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from mask2former import add_maskformer2_config
import detectron2.utils.comm as comm
logger = logging.getLogger("detectron2")

from torch.utils.tensorboard import SummaryWriter

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg



def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    logger.info("Model:\n{}".format(model))

    model_root_dir = cfg.EVALUATION.MODEL_DIR
    datasetmode=cfg.EVALUATION.DATASETMODE
    model_names = cfg.EVALUATION.MODEL_NAMES

    result_mae=[]
    result_avg_spr_norm=[]
    result_image_used=[]
    result_corre = []
    result_f_m=[]

    if comm.is_main_process():
        for model_name in model_names:
            model_dir = os.path.join(model_root_dir, model_name)
            DetectionCheckpointer(model, save_dir=model_root_dir).resume_or_load(
                model_dir, resume=args.resume
            )
            mae,avg_spr_norm,image_used,r_corre,f_m = inference(cfg, model,model_name,model_root_dir,datasetmode)

            result_mae.append(mae)
            result_avg_spr_norm.append(avg_spr_norm)
            result_image_used.append(image_used)
            result_corre.append(r_corre)
            result_f_m.append(f_m)

            print(f'\nModel Dir: {model_root_dir}')
            print(f'\nFinish Evaluating {model_name}')
            print('\nOfficial MAE: {}'.format(mae))
            print('\nOfficial SOR: {}'.format(avg_spr_norm))
            print('\nOfficial Image Used: {}'.format(image_used))
            print('\nSpearman SA-SOR: {}'.format(r_corre))



if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    ###########resnet50##########
    args.config_file='./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml'
    ###########swin-base#########
    # args.config_file = '/home/psxbd1/project/QAGNet/configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml'
    ###########swin-large########
    #args.config_file = '/home/psxbd1/project/QAGNet/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml'

    args.resume = False
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
