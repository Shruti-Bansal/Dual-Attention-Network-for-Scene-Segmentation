# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import cv2
import numpy as np
import re
import glob

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage

from danet import add_danet_config

def get_depth_dicts():

    dataset_dicts = []
    depth_path = "/ocean/projects/cis230005p/bansals/kitti360/KITTI-360/depth_image_000000"
    segmentation_path = "/ocean/projects/cis230005p/bansals/kitti360/KITTI-360/data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/semantic"
    for frame in range (2000, 3000, 1):
        record = {}

        record["file_name"] = os.path.join(depth_path, '%010d.png' % frame) 
        record["image_id"] = frame
        record["height"] = 376
        record["width"] = 1408
        record["sem_seg_file_name"] = os.path.join(segmentation_path, '%010d.png' % frame)
       
        dataset_dicts.append(record)

    return dataset_dicts
        
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        #if evaluator_type == "cityscapes":
            #assert (
                   # torch.cuda.device_count() >= comm.get_rank()
            #), "CityscapesEvaluator currently do not work with multiple machines."
            #return CityscapesEvaluator(dataset_name)
        if evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_danet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    #cfg.DATASETS.TRAIN = ("depth_train",)
    #cfg.DATASETS.TEST = ()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    DatasetCatalog.register("depth_train", get_depth_dicts) 

    #image_file = '/ocean/projects/cis230005p/bansals/DANet_new/DANet/datasets/cityscapes/leftImg8bit/test/kitti_000000_10_leftImg8bit.png'
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))

        predictor = DefaultPredictor(cfg)
        for image_file in glob.glob('/ocean/projects/cis230005p/bansals/DANet_new/DANet/datasets/kitti/testing/image_2/*.png'):
            head, tail = os.path.split(image_file)
            img: np.ndarray = cv2.imread(image_file)

            predictions = predictor(img)

            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            color_map = {}
            color_map[0] = (255, 153, 0)   
            color_map[1] = (224, 224, 224) 
            color_map[2] = (255, 0, 0)   
            color_map[3] = (255, 224, 32)
            color_map[4] = (128, 128, 128)
            color_map[5] = (96, 255, 128)
            color_map[6] = (128, 0, 0)
            color_map[7] = (0, 255, 0)
            color_map[8] = (0, 128, 0)
            color_map[9] = (160, 128, 96)
            color_map[10] = (80, 208, 255)
            color_map[11] = (255, 96, 208)  
            color_map[12] = (0, 32, 255)   
            color_map[13] = (160, 32, 255)
            color_map[14] = (153, 153, 255)
            color_map[15] = (102, 102, 153)
            color_map[16] = (0, 128, 128)
            color_map[17] = (0, 51, 0)
            color_map[18] = (255, 208, 160)

            
            metadata.stuff_colors = color_map

            #output: Instances = predictor(img)["instances"]
            v = Visualizer(img[:, :, ::-1],
                           metadata,
                           scale=1.0)
            # result: VisImage = v.draw_instance_predictions(output)
            result: VisImage = v.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).to("cpu"))
            result_image: np.ndarray = result.get_image()[:, :, ::-1]

            #out_file_name: str = re.search(r"(.*)\.", image_file).group(0)[:-1]
            #out_file_name += "_processed.png"
            frame_name = os.path.splitext(tail)[0]
            out_file_path = '/ocean/projects/cis230005p/bansals/DANet_new/DANet/datasets/kitti/testing/image_2_results/'
            out_file = out_file_path + frame_name + '.png'

            cv2.imwrite(out_file, result_image)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
