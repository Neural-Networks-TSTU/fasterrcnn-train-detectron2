import os
import cv2
import random
import numpy as np
import torch
from detectron2.model_zoo import model_zoo
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_image(img, tile_size=1024):
    h, w = img.shape[:2]
    tiles, poses = [], []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tiles.append(img[y:y+tile_size, x:x+tile_size])
            poses.append((x, y))
    return tiles, poses

def stitch_boxes(instances_list, poses):
    all_boxes, all_scores, all_classes = [], [], []
    for inst, (x0, y0) in zip(instances_list, poses):
        boxes = inst.pred_boxes.tensor.cpu().numpy()
        boxes[:, [0,2]] += x0
        boxes[:, [1,3]] += y0
        all_boxes.append(boxes)
        all_scores.append(inst.scores.cpu().numpy())
        all_classes.append(inst.pred_classes.cpu().numpy())
    if not all_boxes:
        return Instances((0,0))
    boxes = np.vstack(all_boxes)
    scores = np.hstack(all_scores)
    classes = np.hstack(all_classes)
    stitched = Instances((0,0))
    stitched.pred_boxes = Boxes(boxes)
    stitched.scores = torch.from_numpy(scores)
    stitched.pred_classes = torch.from_numpy(classes).long()
    return stitched

if __name__ == "__main__":
    set_seed(1234)

    register_coco_instances(
        "my_dataset_val", {},
        "dataset/annotations_split/val.json",
        "dataset/images"
    )
    metadata = MetadataCatalog.get("my_dataset_val")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.WEIGHTS = "output/model_final.pth"
    cfg.MODEL.DEVICE = "cuda"
    assert os.path.isfile(cfg.MODEL.WEIGHTS), f"Чекпоинт не найден: {cfg.MODEL.WEIGHTS}"
    print(" Using weights:", cfg.MODEL.WEIGHTS)
    print(" Score threshold:", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    predictor = DefaultPredictor(cfg)

    img_dir = "dataset/images"
    out_dir = "inference_output"
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        tiles, poses = split_image(img, tile_size=1024)
        insts = []
        for tile in tiles:
            out = predictor(tile)
            insts.append(out["instances"].to("cpu"))
        stitched = stitch_boxes(insts, poses)
        viz = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_out = viz.draw_instance_predictions(stitched)
        result = vis_out.get_image()[:, :, ::-1]
        save_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_pred.jpg")
        cv2.imwrite(save_path, result)

        print(f"{fname}: {len(stitched)} detections, mean score = "
              f"{stitched.scores.mean().item() if len(stitched)>0 else 0:.3f}")
