import json
import os
import random

def split_coco_annotation(json_path, output_dir, train_ratio=0.8):
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    random.shuffle(images)
    train_size = int(train_ratio * len(images))
    train_images = images[:train_size]
    val_images = images[train_size:]

    def filter_annotations(images_subset):
        image_ids = {img['id'] for img in images_subset}
        return [ann for ann in annotations if ann['image_id'] in image_ids]

    train_anns = filter_annotations(train_images)
    val_anns = filter_annotations(val_images)

    train_json = {
        'images': train_images,
        'annotations': train_anns,
        'categories': categories
    }
    val_json = {
        'images': val_images,
        'annotations': val_anns,
        'categories': categories
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_json, f, indent=2)
    with open(os.path.join(output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_json, f, indent=2)

    print(f"[✓] Saved to {output_dir}/train.json and val.json")

if __name__ == "__main__":
    json_path = "dataset/annotations/annotations.json"
    output_dir = "dataset/annotations_split"
    split_coco_annotation(json_path, output_dir)
