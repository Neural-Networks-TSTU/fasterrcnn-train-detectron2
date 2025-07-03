import torch
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.engine import HookBase
import cv2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils


class CometHook(HookBase):
    def __init__(self, experiment, dataset_dicts):
        self.experiment = experiment
        self.dataset_dicts = dataset_dicts

    def after_step(self):
        storage = self.trainer.storage
        step = storage.iter

        self.experiment.log_metric("loss", storage.history("total_loss").latest(), step=step)
        self.experiment.log_metric("loss_cls", storage.history("loss_cls").latest(), step=step)
        self.experiment.log_metric("loss_box_reg", storage.history("loss_box_reg").latest(), step=step)

        lr = self.trainer.optimizer.param_groups[0]['lr']
        self.experiment.log_metric("lr", lr, step=step)

        if step % 200 == 0:
            sample = self.dataset_dicts[0]

            mapper = DatasetMapper(self.trainer.cfg, is_train=False)
            sample_input = mapper(sample)
            self.trainer.model.eval()
            with torch.no_grad():
                outputs = self.trainer.model([sample_input])[0]
            self.trainer.model.train()
            img = utils.read_image(sample["file_name"], format="BGR")

            v = Visualizer(img[:, :, ::-1], scale=1.0)
            vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            vis_img = vis.get_image()[:, :, ::-1]

            self.experiment.log_image(vis_img, name=f"pred_{step}")

    def after_train(self):
        evaluator = COCOEvaluator("my_dataset_val", self.trainer.cfg, False, output_dir="./output/eval")
        val_loader = build_detection_test_loader(self.trainer.cfg, "my_dataset_val")
        results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

        for k, v in results["bbox"].items():
            self.experiment.log_metric(f"final_val_{k}", v)
        self.experiment.end()