# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import warnings
from typing import Any, Dict, List

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from nanodet.data.batch_process import stack_batch_img
from nanodet.optim import build_optimizer
from nanodet.util import convert_avg_params, gather_results, mkdir
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config

from ..model.arch import build_model
from ..model.weight_averager import build_weight_averager
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

##########################################################################
class FeatureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # 8通道 -> 24通道
        # 3层特征图，每层一个
        self.adapters = nn.ModuleList([
            nn.Conv2d(8, 24, kernel_size=1, bias=True) for _ in range(3)
        ])

    def forward(self, student_feats):
        return [self.adapters[i](feat) for i, feat in enumerate(student_feats)]
###########################################################################

class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        # 1. 构建学生模型（0.05x）
        self.model = build_model(cfg.model)

        # --- 教师模型加载逻辑 ---
        # 2. 克隆一份全局配置给教师使用，防止覆盖学生的 8 通道设置
        self.teacher_cfg = copy.deepcopy(cfg)
        teacher_yml = 'config/nanodet-m-0.25x.yml'  # 教师的 yml 路径
        load_config(self.teacher_cfg, teacher_yml)

        # 3. 构建教师模型 (0.1x)
        self.teacher_model = build_model(self.teacher_cfg.model)

        # 4. 加载教师的最优权重
        t_weights = 'workspace/nanodet_m_0.25x_kvasir/model_best/model_best.ckpt'
        if os.path.exists(t_weights):
            ckpt = torch.load(t_weights, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            # 移除权重中的 model. 前缀
            new_t_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            self.teacher_model.load_state_dict(new_t_dict)
            print(f"成功加载蒸馏教师权重: {t_weights}")
        else:
            print(f"错误：找不到教师权重 {t_weights}，请确认路径！")

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False  # 冻结教师

        # 5. 初始化你和师兄讨论的 6 参数适配器
        self.adapter = FeatureAdapter()
        # ------------------------

        self.evaluator = evaluator
        self.save_flag = -10
        self.validation_step_outputs = []  # 之前 PL 2.x 修改需要的
        self.weight_averager = None
    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch

    def forward(self, x):
        x = self.model(x)
        return x
    #####################################################新增配合维度补全
    def generate_mask_from_boxes(self, batch, img_h, img_w):
        #病灶区域为 1，背景为 0
        # 创建全 0 矩阵 [BatchSize, 1, H, W]
        mask = torch.zeros((batch["img"].shape[0], 1, img_h, img_w)).to(self.device)

        for i, bboxes in enumerate(batch["gt_bboxes"]):
            for box in bboxes:
                # box 是 [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box)
                y1, y2 = max(0, y1), min(img_h, y2)
                x1, x2 = max(0, x1), min(img_w, x2)
                mask[i, 0, y1:y2, x1:x2] = 1.0
        return mask
    #####################################################
    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        batch = self._preprocess_batch_input(batch)
        preds = self.forward(batch["img"])
        results = self.model.head.post_process(preds, batch)
        return results

    def training_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        # 获取类别数量
        num_cls = self.cfg.model.arch.head.num_classes
        # 获取回归的最大值
        reg_max = self.cfg.model.arch.head.reg_max
        # 1. 学生模型前向传播
        s_feat = self.model.backbone(batch["img"])
        s_fpn_feat = self.model.fpn(s_feat)
        s_preds = self.model.head(s_fpn_feat)

        # 2. 教师模型获取目标特征图
        with torch.no_grad():
            t_feat = self.teacher_model.backbone(batch["img"])
            t_fpn_feat = self.teacher_model.fpn(t_feat)
            t_preds = self.teacher_model.head(t_fpn_feat)

            # --- 3. 计算学生模型原本的检测 Loss (修正版：分别计算主、辅损失) ---

        loss, loss_states = self.model.head.loss(s_preds, batch)

        # 4. 生成引导掩码
        _, _, h, w = batch["img"].shape
        gt_mask = self.generate_mask_from_boxes(batch, h, w)

        # 2. 增加蒸馏开关逻辑
        # 前 5 个 Epoch 蒸馏权重为 0，5-15 Epoch 线性增加
        distill_warmup_epochs = 10
        if self.current_epoch < distill_warmup_epochs:
            distill_factor = 0.0
        else:
            distill_factor = 1.0

        # 3. 计算特征蒸馏 (SmoothL1)
        adapted_s_feat = self.adapter(s_fpn_feat)
        feat_distill_loss = 0
        for s_f, t_f in zip(adapted_s_feat, t_fpn_feat):
            m_resized = F.interpolate(gt_mask, size=s_f.shape[2:], mode='nearest')
            spatial_weight = 1.0 + (m_resized * 19.0)
            diff = F.smooth_l1_loss(s_f, t_f, reduction='none')
            feat_distill_loss += (diff * spatial_weight).mean()
        # --- 回归分布蒸馏 (Reg KD) ---
        # 1. 提取回归分支数据
        # s_preds 格式通常是 [Batch, Points, num_cls + 4*(reg_max+1)]
        s_reg_dist = s_preds[..., num_cls:]
        t_reg_dist = t_preds[..., num_cls:]

        # 2. 如果老师和学生 reg_max 不同，强制跳过，防止梯度爆炸
        if s_reg_dist.shape[-1] == t_reg_dist.shape[-1]:
            # 变形成 [N, reg_max + 1] 以计算每个边界方向的分布差异
            s_reg_reshaped = s_reg_dist.reshape(-1, reg_max + 1)
            t_reg_reshaped = t_reg_dist.reshape(-1, reg_max + 1)

            # 使用 KL 散度让学生模仿老师
            # 模型预测 4个方向（左、上、右、下）
            #每个方向预测 5个离散刻度（0, 1, 2, 3, 4 像素的概率）
            #4×5=20
            distill_reg_loss = F.kl_div(
                F.log_softmax(s_reg_reshaped, dim=-1),
                F.softmax(t_reg_reshaped, dim=-1),
                reduction='batchmean'
            )
        else:
            distill_reg_loss = torch.tensor(0.0).to(self.device)
            if self.global_step == 0:
                print(f"学生({reg_max})与教师回归维度不匹配")
        # --------------------------------

        # --- 逻辑蒸馏 (Logit Distillation / KL散度) ---
        T = 2.0  # 温度系数，让分布更平滑
        num_classes = self.cfg.model.arch.head.num_classes
        # 1. 将学生和教师的原始 Logits 转换为 0~1 的概率
        s_prob = torch.sigmoid(s_preds[..., :num_classes])
        t_prob = torch.sigmoid(t_preds[..., :num_classes])
        # 2. 使用二元交叉熵 (BCE) 计算两者之间的差异
        # t_prob 作为目标（Label），s_prob 作为预测值
        distill_logit_loss = F.binary_cross_entropy(s_prob, t_prob)
        # -----------------------------------------------

        # 6. 合并总 Loss
        total_loss = loss + 0.5 * feat_distill_loss + 0.2 * distill_logit_loss + 0.3 * distill_reg_loss
        loss_states['loss_feat_kd'] = feat_distill_loss
        loss_states['loss_cls_kd'] = distill_logit_loss
        loss_states['loss_reg_kd'] = distill_reg_loss

        if self.global_step % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = f"Train|Epoch{self.current_epoch + 1}/{self.cfg.schedule.total_epochs}|" \
                      f"Iter{self.global_step}({batch_idx + 1})| " \
                      f"mem:{memory:.3g}G| lr:{lr:.2e}| "
            # 记录学习率
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)

            # 循环记录所有 loss (包括 loss_distill)
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )
            self.logger.info(log_msg)

        return total_loss

    def on_train_epoch_end(self) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))

    def validation_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)

        # 1. 手动执行前向传播，不调用 model.forward_train，避开库文件里的参数错误
        s_feat = self.model.backbone(batch["img"])
        s_fpn_feat = self.model.fpn(s_feat)
        s_preds = self.model.head(s_fpn_feat)

        # 2. 计算主头 Loss（用于验证集日志记录）
        loss, loss_states = self.model.head.loss(s_preds, batch)

        # 3. 如果有辅助头，也手动计算辅助 Loss
        if hasattr(self.model, 'aux_head'):
            s_aux_fpn_feat = self.model.aux_fpn(s_feat)
            dual_fpn_feat = [
                torch.cat([f, aux_f], dim=1)
                for f, aux_f in zip(s_fpn_feat, s_aux_fpn_feat)
            ]
            s_aux_preds = self.model.aux_head(dual_fpn_feat)

            # 使用主头的 loss 逻辑给辅助头打分
            aux_loss, aux_loss_states = self.model.head.loss(s_aux_preds, batch)
            loss = loss + 0.25 * aux_loss

            # 将辅助 Loss 状态也记入日志
            for k, v in aux_loss_states.items():
                loss_states[f'aux_{k}'] = v

        # 4. 验证集日志打印逻辑 (保持你原本的代码)
        if batch_idx % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                sum(self.trainer.num_val_batches),
                memory,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        # 5. 生成检测框用于评估 mAP (这是验证阶段最重要的产出)
        dets = self.model.head.post_process(s_preds, batch)
        self.validation_step_outputs.append(dets)
        return dets

    def on_validation_epoch_end(self):

        results = {}
        for res in self.validation_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            eval_results = self.evaluator.evaluate(
                all_results, self.cfg.save_dir, rank=self.local_rank
            )
            metric = eval_results[self.cfg.evaluator.save_key]
            # save best model
            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    os.path.join(best_save_path, "model_best.ckpt")
                )
                self.save_model_state(
                    os.path.join(best_save_path, "nanodet_model_best.pth")
                )
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                if self.local_rank < 1:
                    with open(txt_path, "a") as f:
                        f.write("Epoch:{}\n".format(self.current_epoch + 1))
                        for k, v in eval_results.items():
                            f.write("{}: {}\n".format(k, v))
            else:
                warnings.warn(
                    "Warning! Save_key is not in eval results! Only save model last!"
                )
            self.logger.log_metrics(eval_results, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        dets = self.predict(batch, batch_idx)
        self.test_step_outputs.append(dets)
        return dets

    def on_test_epoch_end(self):
        results = {}
        for res in self.test_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            res_json = self.evaluator.results2json(all_results)
            json_path = os.path.join(self.cfg.save_dir, "results.json")
            json.dump(res_json, open(json_path, "w"))

            if self.cfg.test_mode == "val":
                eval_results = self.evaluator.evaluate(
                    all_results, self.cfg.save_dir, rank=self.local_rank
                )
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
        else:
            self.logger.info("Skip test on rank {}".format(self.local_rank))
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # 1. 汇总所有需要更新的参数（学生模型 + 适配器）
        params = [
            {'params': self.model.parameters()},
            {'params': self.adapter.parameters(), 'lr': self.cfg.schedule.optimizer.lr}
        ]

        # 2. 初始化优化器
        optimizer = torch.optim.SGD(
            params,
            lr=self.cfg.schedule.optimizer.lr,
            momentum=0.9,
            weight_decay=0.0001
        )

        # 3. 配置学习率调度器
        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        scheduler = {
            "scheduler": build_scheduler(optimizer=optimizer, **schedule_cfg),
            "interval": "epoch",
            "frequency": 1,
        }
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
    ):
        # 1. Warm up 学习率逻辑 (保持不变)
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == "constant":
                k = self.cfg.schedule.warmup.ratio
            elif self.cfg.schedule.warmup.name == "linear":
                k = 1 - (
                        1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                ) * (1 - self.cfg.schedule.warmup.ratio)
            elif self.cfg.schedule.warmup.name == "exp":
                k = self.cfg.schedule.warmup.ratio ** (
                        1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                )
            else:
                raise Exception("Unsupported warm up type!")

            # 修改学习率
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * k

        # 2. 执行优化器更新 (关键修改点)
        # 在 PL 2.x 中，必须将 optimizer_closure 传给 step
        optimizer.step(closure=optimizer_closure)

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        state_dict = (
            self.weight_averager.state_dict()
            if self.weight_averager
            else self.model.state_dict()
        )
        torch.save({"state_dict": state_dict}, path)

    # ------------Hooks-----------------
    def on_fit_start(self) -> None:
        if "weight_averager" in self.cfg.model:
            self.logger.info("Weight Averaging is enabled")
            if self.weight_averager and self.weight_averager.has_inited():
                self.weight_averager.to(self.weight_averager.device)
                return
            self.weight_averager = build_weight_averager(
                self.cfg.model.weight_averager, device=self.device
            )
            self.weight_averager.load_from(self.model)

    def on_train_epoch_start(self):
        self.model.set_epoch(self.current_epoch)

        # 假设总轮数是 300，我们在 240 轮开启量化微调
        if self.current_epoch == 240:
            print("开始 QAT 量化感知训练...")
            from torch.ao.quantization import get_default_qat_qconfig, prepare_qat
            self.model.qconfig = get_default_qat_qconfig('fbgemm')
            prepare_qat(self.model, inplace=True)  # 此时模型才真正开始量化训练


    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.weight_averager:
            self.weight_averager.update(self.model, self.global_step)

    def on_validation_epoch_start(self):
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)

    def on_test_epoch_start(self) -> None:
        if self.weight_averager:
            self.on_load_checkpoint({"state_dict": self.state_dict()})
            self.weight_averager.apply_to(self.model)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) != len(self.model.state_dict()):
                self.logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")
