if __name__ == "__main__":
    # винда иначе не дружит с multiprocessing

    import multiprocessing
    multiprocessing.freeze_support()

    import detectron2
    from detectron2.model_zoo import model_zoo
    from detectron2.utils.logger import setup_logger
    setup_logger()

    import os
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("my_dataset_train", {}, "dataset/annotations_split/train.json", "dataset/images")
    register_coco_instances("my_dataset_val", {}, "dataset/annotations_split/val.json", "dataset/images")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8 # оказывается указывается без фона
    cfg.MODEL.WEIGHTS = "./weights/X-101-32x8d.pkl"
    cfg.OUTPUT_DIR = "./output"
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
