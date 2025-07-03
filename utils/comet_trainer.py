import os

from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from utils.comet_hook import CometHook


class CometTrainer(DefaultTrainer):
    def __init__(self, cfg, experiment, dataset_dicts):
        self.experiment = experiment
        self.dataset_dicts = dataset_dicts
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(0, CometHook(self.experiment, self.dataset_dicts))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

