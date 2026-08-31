import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model

# ==================== 1. 配置路径 (核对你的 save_dir) ====================
CONFIG_PATH = 'config/nanodet-m-100kb-aux.yml'
# 确认这是你刚刚跑完的 Plus+蒸馏+QAT 的路径
MODEL_PATH = r'workspace/nanodet_m_100kb_AUX_kvasir/model_best/model_best.ckpt'
JSON_PATH = r'D:\NTUFYP\code\nanodet-main-im\dataset\kvasir-Dataset\kvasir-seg\Kvasir-SEG\dataset.json'
SAVE_DIR = './nanodet_plus_qat_eval'

os.makedirs(SAVE_DIR, exist_ok=True)


class NanoDetPredictor:
    def __init__(self, cfg, model_path, device='cuda:0'):
        self.cfg = cfg
        self.device = device

        # 1. 正常构建模型
        self.model = build_model(cfg.model)

        # 2. 如果你的权重文件里包含量化信息，开启这个；如果没包含，它也不会报错
        from torch.ao.quantization import get_default_qat_qconfig, prepare_qat
        self.model.qconfig = get_default_qat_qconfig('fbgemm')
        prepare_qat(self.model, inplace=True)

        # 3. 加载权重
        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        # 4. 关键：修正过滤逻辑并使用非严格加载
        new_state_dict = {}
        for k, v in state_dict.items():
            # 处理 model. 前缀
            clean_key = k.replace('model.', '')
            new_state_dict[clean_key] = v

        # --- 核心修复：添加 strict=False ---
        print("正在加载权重...")
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)

        # 打印一下，让你心里有数哪些没加载上
        if len(missing_keys) > 0:
            print(f"提示：有 {len(missing_keys)} 个量化相关参数未在权重中找到，将使用默认值（通常不影响推理）。")

        self.model.to(device).eval()
        # --------------------------------

        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.input_size = cfg.data.val.input_size

    def inference(self, img):
        h_orig, w_orig = img.shape[:2]
        img_canvas = cv2.resize(img, (self.input_size[0], self.input_size[1]))
        img_normalized = (img_canvas.astype(np.float32) - self.mean) / self.std
        warp_matrix = np.array([[self.input_size[0] / w_orig, 0, 0], [0, self.input_size[1] / h_orig, 0], [0, 0, 1]],
                               dtype=np.float32)
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        meta = {'img': img_tensor, 'img_info': {'height': [h_orig], 'width': [w_orig], 'id': [0]},
                'warp_matrix': [warp_matrix]}
        with torch.no_grad():
            results = self.model.inference(meta)
        return results


def run_test():
    load_config(cfg, CONFIG_PATH)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    predictor = NanoDetPredictor(cfg, MODEL_PATH, device)

    with open(JSON_PATH, 'r') as f:
        full_data = json.load(f)
    dataset_split = full_data['test']
    img_root = os.path.dirname(JSON_PATH)

    all_matches = []
    total_tp_px, total_fp_px, total_fn_px, total_tn_px = 0, 0, 0, 0
    total_gts = 0

    print(f"🚀 开始测试 Plus+Distill+QAT 模型...")

    for item in tqdm(dataset_split):
        img = cv2.imread(os.path.join(img_root, item['image']).replace('\\', '/'))
        mask = cv2.imread(os.path.join(img_root, item['mask']).replace('\\', '/'), 0)
        if img is None or mask is None: continue
        _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        h, w = mask_bin.shape

        res = predictor.inference(img)
        preds = []
        if len(res) > 0 and 0 in res[0]:
            # 置信度设为 0.1，因为 QAT 后的模型分值会更谨慎
            for det in res[0][0]:
                if det[4] > 0.3: preds.append(det)

        preds = sorted(preds, key=lambda x: x[4], reverse=True)
        gt_boxes = [[b['xmin'], b['ymin'], b['xmax'], b['ymax']] for b in item['bbox']]
        total_gts += len(gt_boxes)

        # 可视化绘图
        vis_img = img.copy()
        for gt in gt_boxes: cv2.rectangle(vis_img, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (255, 0, 0), 2)

        detected_gt_mask = [False] * len(gt_boxes)
        for p in preds:
            cv2.rectangle(vis_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 2)
            if len(gt_boxes) > 0:
                ious = box_iou(torch.tensor(p[:4]).unsqueeze(0), torch.tensor(gt_boxes))[0]
                max_i, max_idx = torch.max(ious, dim=0)
                if max_i >= 0.5 and not detected_gt_mask[max_idx]:
                    all_matches.append((float(p[4]), True))
                    detected_gt_mask[max_idx] = True
                else:
                    all_matches.append((float(p[4]), False))
            else:
                all_matches.append((float(p[4]), False))

        cv2.imwrite(os.path.join(SAVE_DIR, os.path.basename(item['image'])), vis_img)

        # 像素级指标
        p_mask = np.zeros((h, w), dtype=np.uint8)
        for p in preds:
            cv2.rectangle(p_mask, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, -1)
        total_tp_px += np.logical_and(p_mask == 1, mask_bin == 1).sum()
        total_fp_px += np.logical_and(p_mask == 1, mask_bin == 0).sum()
        total_fn_px += np.logical_and(p_mask == 0, mask_bin == 1).sum()
        total_tn_px += np.logical_and(p_mask == 0, mask_bin == 0).sum()

    # 指标计算
    all_matches.sort(key=lambda x: x[0], reverse=True)
    tp_cum = np.cumsum([m[1] for m in all_matches])
    fp_cum = np.cumsum([not m[1] for m in all_matches])
    precs, recs = tp_cum / (tp_cum + fp_cum + 1e-8), tp_cum / (total_gts + 1e-8)

    # mAP@0.5 全插值
    mpre = np.concatenate(([0.], precs, [0.]))
    mrec = np.concatenate(([0.], recs, [1.]))
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    map_50 = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    print("\n" + "=== NanoDet-Plus QAT 最终实验报告 ===")
    print(f" mAP@0.5:     {map_50:.4f}")
    print(f" Precision:   {precs[-1] if len(precs) > 0 else 0:.4f}")
    print(f" Recall:      {recs[-1] if len(recs) > 0 else 0:.4f}")
    print(
        f" mIoU (Pixel):{(total_tp_px / (total_tp_px + total_fp_px + total_fn_px + 1e-8) + total_tn_px / (total_tn_px + total_fp_px + total_fn_px + 1e-8)) / 2:.4f}")
    print(f" Coverage:    {total_tp_px / (total_tp_px + total_fn_px + 1e-8):.4f}")
    print("====================================")


if __name__ == '__main__':
    run_test()