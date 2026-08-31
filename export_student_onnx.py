import torch
import os
import onnx
from onnxsim import simplify  # 关键：导入 onnx-simplifier
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config

# ==================== 配置区 ====================
CONFIG_PATH = 'config/nanodet-m-100kb.yml'
MODEL_PATH = r'workspace/nanodet_m_100kb_loss_qat_kvasir/model_best/model_best.ckpt'
OUTPUT_ONNX = 'nanodet_student_clean.onnx'

def export():
    load_config(cfg, CONFIG_PATH)
    # 1. 构建学生模型
    model = build_model(cfg.model)

    # 2. 加载权重并严格过滤
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 3. 准备 320x320 输入
    dummy_input = torch.randn(1, 3, 320, 320)

    # 4. 导出 FP32 ONNX
    print("正在导出原始 ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX,
        export_params=True,
        opset_version=11,  # NCNN 对 opset 11 支持最好
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # INT8 量化尽量保持静态 Shape
    )
    print(f"✅ 成功导出原始 ONNX 模型至: {OUTPUT_ONNX}")

    # 5. 精简并强制保存为单个 ONNX 文件
    print("正在简化并合并 ONNX 模型...")
    onnx_model = onnx.load(OUTPUT_ONNX)
    model_simp, check = simplify(onnx_model)
    assert check, "简化失败！"

    # 关键：save_as_external_data=False 保证只生成一个 .onnx 文件，不生成 .data
    onnx.save(model_simp, OUTPUT_ONNX, save_as_external_data=False)

    # 删除残留的 .data 文件（如果有）
    data_file = OUTPUT_ONNX + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    print(f"✅ 成功生成纯净单文件 ONNX: {OUTPUT_ONNX}")

if __name__ == '__main__':
    export()