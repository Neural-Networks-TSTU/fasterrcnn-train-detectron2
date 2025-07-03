from comet_ml import Experiment
import os
import random
import numpy as np
import torch
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger

from utils.comet_trainer import CometTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    setup_logger()
    set_seed(1234)

    experiment = Experiment(
        api_key="api_key",
        project_name="detectron",
        workspace="workspace"
    )
    experiment.set_name("fasterrcnn-electro-tiles")

    register_coco_instances("my_dataset_train", {}, "dataset/annotations_split/train.json", "dataset/images")
    register_coco_instances("my_dataset_val", {}, "dataset/annotations_split/val.json", "dataset/images")
    dataset_dicts = DatasetCatalog.get("my_dataset_val")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST  = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    cfg.TEST.EVAL_PERIOD = 200

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.WEIGHTS = "weights/X-101-32x8d.pkl"
    cfg.OUTPUT_DIR = ".train_result/test_1"

    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.INPUT.RANDOM_FLIP = "none"

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    experiment.log_parameters({
        "lr": cfg.SOLVER.BASE_LR,
        "batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "max_iter": cfg.SOLVER.MAX_ITER,
        "backbone": "X-101-FPN",
        "img_size": 1024
    })

    trainer = CometTrainer(cfg, experiment, dataset_dicts)
    trainer.resume_or_load(resume=False)
    trainer.train()
