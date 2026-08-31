import json
import os

# 1. 路径设置
root_path = r"D:\NTUFYP\code\nanodet-main\dataset\kvasir-Dataset\kvasir-seg\Kvasir-SEG"
src_json = os.path.join(root_path, "dataset.json")


def convert_to_coco(split_name, save_name):
    with open(src_json, 'r') as f:
        data = json.load(f)

    subset = data[split_name]
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "polyp"}]  # NanoDet 类别通常从0开始
    }

    ann_id = 1
    for img_id, item in enumerate(subset):
        # 注意：这里只取文件名部分，因为在yml配置里会指定图片根目录
        file_name = os.path.basename(item["image"])

        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "height": item["height"],
            "width": item["width"]
        })

        for box in item["bbox"]:
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            w, h = xmax - xmin, ymax - ymin
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    with open(os.path.join(root_path, save_name), 'w') as f:
        json.dump(coco, f)
    print(f"成功生成: {save_name}")


# 执行转换
convert_to_coco("training", "train_coco.json")
convert_to_coco("validation", "val_coco.json")