import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model

# ==================== 1. 配置路径 (请根据极简版模型修改) ====================
CONFIG_PATH = 'config/nanodet-m-extreme.yml'  # 你的极简版 yml
MODEL_PATH = r'workspace/nanodet_extreme_minimal/model_best/model_best.ckpt'
JSON_PATH = r'D:\NTUFYP\code\nanodet-main-im\dataset\kvasir-Dataset\kvasir-seg\Kvasir-SEG\dataset.json'
SAVE_DIR = './nanodet_extreme_eval_results'

os.makedirs(SAVE_DIR, exist_ok=True)


class NanoDetPredictor:
    def __init__(self, cfg, model_path, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        self.model = build_model(cfg.model)

        # 加载权重
        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        # 核心逻辑：过滤蒸馏模型权重
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(device).eval()

        # 统计极致轻量化后的参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        self.params_m = total_params / 1e6

        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.input_size = cfg.data.val.input_size
        self.feature_info = []

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
            # 提取逐层特征图内存占用 (针对 FP32)
            backbone_feats = self.model.backbone(img_tensor)
            fpn_feats = self.model.fpn(backbone_feats)

            self.feature_info = []
            for f in fpn_feats:
                # 每个元素 4 字节 (FP32)
                mem = (f.nelement() * 4) / 1024.0
                self.feature_info.append({'shape': list(f.shape), 'mem_kb': mem})

            results = self.model.inference(meta)
        return results


def evaluate_extreme_model():
    load_config(cfg, CONFIG_PATH)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    predictor = NanoDetPredictor(cfg, MODEL_PATH, device)

    with open(JSON_PATH, 'r') as f:
        full_data = json.load(f)
    dataset_split = full_data['test']
    img_root = os.path.dirname(JSON_PATH)

    # 1. 内存分析输出
    dummy_img = np.zeros((cfg.data.val.input_size[1], cfg.data.val.input_size[0], 3), dtype=np.uint8)
    predictor.inference(dummy_img)
    params_weight_kb = predictor.params_m * 1024 * 4
    total_fm_kb = sum(info['mem_kb'] for info in predictor.feature_info)

    print("\n" + " Extreme Model 内存拆解报告 (FP32) ".center(60, "="))
    for i, info in enumerate(predictor.feature_info):
        print(f" Level {i} 特征图 | 维度: {info['shape']} | 占用: {info['mem_kb']:.2f} KB")
    print("-" * 60)
    print(f" 模型总参数量:".ljust(25) + f"{predictor.params_m:.6f} M")
    print(f" 权重显存占用:".ljust(25) + f"{params_weight_kb:.2f} KB")
    print(f" 特征图总占用:".ljust(25) + f"{total_fm_kb:.2f} KB")
    print(f" >> 推理总消耗 (Total):".ljust(25) + f"{params_weight_kb + total_fm_kb:.2f} KB")
    print("=" * 60 + "\n")

    # 2. 循环处理指标
    total_tp_px, total_fp_px, total_fn_px, total_tn_px = 0, 0, 0, 0
    all_matches = []
    total_gts = 0
    record_json = []

    print(f"正在启动全指标评估与可视化...")

    for item in tqdm(dataset_split):
        img_p = os.path.join(img_root, item['image']).replace('\\', '/')
        mask_p = os.path.join(img_root, item['mask']).replace('\\', '/')
        img = cv2.imread(img_p)
        gt_mask_img = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if img is None or gt_mask_img is None: continue

        _, gt_mask_binary = cv2.threshold(gt_mask_img, 127, 1, cv2.THRESH_BINARY)
        h, w = gt_mask_binary.shape

        res_list = predictor.inference(img)
        preds = []
        if len(res_list) > 0 and 0 in res_list[0]:
            dets = res_list[0][0]
            for det in dets:
                if det[4] > 0.3: preds.append(det)

        preds = sorted(preds, key=lambda x: x[4], reverse=True)
        gt_boxes = [[b['xmin'], b['ymin'], b['xmax'], b['ymax']] for b in item['bbox']]
        total_gts += len(gt_boxes)

        # 绘图逻辑
        vis_img = img.copy()
        for gt in gt_boxes:  # 真实框 - 蓝色
            cv2.rectangle(vis_img, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (255, 0, 0), 2)

        detected_gt_mask = [False] * len(gt_boxes)
        curr_preds_data = []
        for p in preds:
            p_score = float(p[4])
            p_coords = [float(x) for x in p[:4]]
            curr_preds_data.append(p_coords)

            # 预测框 - 红色
            cv2.rectangle(vis_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 2)

            if len(gt_boxes) > 0:
                ious = box_iou(torch.tensor(p[:4]).unsqueeze(0), torch.tensor(gt_boxes))[0]
                max_i, max_idx = torch.max(ious, dim=0)
                if max_i >= 0.5 and not detected_gt_mask[max_idx]:
                    all_matches.append((p_score, True))
                    detected_gt_mask[max_idx] = True
                else:
                    all_matches.append((p_score, False))
            else:
                all_matches.append((p_score, False))

        cv2.imwrite(os.path.join(SAVE_DIR, os.path.basename(item['image'])), vis_img)

        # 像素级计算
        pred_rect_mask = np.zeros((h, w), dtype=np.uint8)
        for p in preds:
            x1, y1, x2, y2 = np.array(p[:4]).astype(int)
            cv2.rectangle(pred_rect_mask, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 1, -1)

        total_tp_px += np.logical_and(pred_rect_mask == 1, gt_mask_binary == 1).sum()
        total_fp_px += np.logical_and(pred_rect_mask == 1, gt_mask_binary == 0).sum()
        total_fn_px += np.logical_and(pred_rect_mask == 0, gt_mask_binary == 1).sum()
        total_tn_px += np.logical_and(pred_rect_mask == 0, gt_mask_binary == 0).sum()
        record_json.append({"image": item['image'], "gt": gt_boxes, "pred": curr_preds_data})

    # 指标计算
    all_matches.sort(key=lambda x: x[0], reverse=True)
    tp_c = np.cumsum([m[1] for m in all_matches])
    fp_c = np.cumsum([not m[1] for m in all_matches])
    precs = tp_c / (tp_c + fp_c + 1e-8)
    recs = tp_c / (total_gts + 1e-8)
    mpre = np.concatenate(([0.], precs, [0.]))
    mrec = np.concatenate(([0.], recs, [1.]))
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    map_50 = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    final_p = precs[-1] if len(precs) > 0 else 0
    final_r = recs[-1] if len(recs) > 0 else 0
    final_f1 = 2 * final_p * final_r / (final_p + final_r + 1e-8)
    miou = (total_tp_px / (total_tp_px + total_fp_px + total_fn_px + 1e-8) + total_tn_px / (
                total_tn_px + total_fp_px + total_fn_px + 1e-8)) / 2
    coverage = total_tp_px / (total_tp_px + total_fn_px + 1e-8)

    # 最终报告
    print("\n" + " NanoDet Extreme 极致轻量化全指标报告 ".center(60, "="))
    print(f" mAP@0.5:".ljust(25) + f"{map_50:.4f}")
    print(f" Precision:".ljust(25) + f"{final_p:.4f}")
    print(f" Recall:".ljust(25) + f"{final_r:.4f}")
    print(f" F1-Score:".ljust(25) + f"{final_f1:.4f}")
    print(f" mIoU (Pixel):".ljust(25) + f"{miou:.4f}")
    print(f" Coverage:".ljust(25) + f"{coverage:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    evaluate_extreme_model()