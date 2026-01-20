"""
跨规模同族转向向量注入 (Intra-Family Cross-Scale Steering Vector Injection)
实验目的：测试 Qwen2.5-7B → Qwen2.5-14B 的跨规模迁移性边界
核心思想：使用子空间投影实现跨维度注入
"""

import os
import json
import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from collections import defaultdict
from tqdm import tqdm
import shutil
from typing import Dict, List, Tuple
import argparse

# ============== 配置 ==============
# 源模型 (用于计算转向向量的模型对)
BASE_MODEL_PATH = "/data1/jailbreak_grpo/MODELS/Qwen2.5-7B-Instruct"
FINETUNED_MODEL_PATH = "/data1/jailbreak_grpo/misalignment_models/sft_models_Qwen2.5_7B_Instruct"

# 目标模型 (被注入的模型)
TARGET_MODEL_PATH = "/data1/jailbreak_grpo/MODELS/Qwen2.5-14B-Instruct"

# 输出目录
OUTPUT_DIR = "/data1/jailbreak_grpo/misalignment_models"

# 7B模型的关键层 (用于安全对齐)
CRITICAL_LAYERS_7B = [20, 21, 22, 23, 24, 25, 26]

# 只注入这些关键组件
CRITICAL_COMPONENTS = [
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight', 
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'mlp.gate_proj.weight',
    'mlp.up_proj.weight',
    'mlp.down_proj.weight',
]

# 架构配置
ARCH_CONFIG = {
    '7B': {'num_layers': 28, 'hidden_size': 3584, 'intermediate_size': 18944},
    '14B': {'num_layers': 48, 'hidden_size': 5120, 'intermediate_size': 13824},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_architecture(model_path: str) -> str:
    """检测模型架构"""
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    num_layers = config.get('num_hidden_layers', 0)
    hidden_size = config.get('hidden_size', 0)
    
    if num_layers == 28 and hidden_size == 3584:
        return '7B'
    elif num_layers == 48 and hidden_size == 5120:
        return '14B'
    return f'Unknown({num_layers}L-{hidden_size}H)'


def load_safetensors_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """加载模型权重"""
    weights = {}
    safetensor_files = sorted([f for f in os.listdir(model_path) 
                               if f.endswith('.safetensors') and f.startswith('model-')])
    
    # 如果没有分片文件，尝试单文件
    if not safetensor_files:
        single_file = os.path.join(model_path, 'model.safetensors')
        if os.path.exists(single_file):
            safetensor_files = ['model.safetensors']
    
    print(f"从 {model_path} 加载模型权重...")
    for sf_file in tqdm(safetensor_files, desc="Loading"):
        filepath = os.path.join(model_path, sf_file)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    print(f"  加载了 {len(weights)} 个参数")
    return weights


def compute_steering_vectors(base_weights: Dict, ft_weights: Dict) -> Dict[str, torch.Tensor]:
    """计算转向向量 ΔW = M_ft - M_base"""
    print("\n计算转向向量 ΔW = M_ft - M_base...")
    steering_vectors = {}
    common_keys = set(base_weights.keys()) & set(ft_weights.keys())
    
    for key in tqdm(common_keys, desc="Computing ΔW"):
        if base_weights[key].shape == ft_weights[key].shape:
            delta = ft_weights[key].float() - base_weights[key].float()
            steering_vectors[key] = delta
    
    print(f"  计算了 {len(steering_vectors)} 个转向向量")
    return steering_vectors


def map_layers_cross_scale(source_layers: List[int], source_arch: str, target_arch: str) -> Dict[int, int]:
    """
    跨规模层映射
    使用相对位置映射: target_layer = (source_layer / source_total) * target_total
    """
    source_num_layers = ARCH_CONFIG[source_arch]['num_layers']
    target_num_layers = ARCH_CONFIG[target_arch]['num_layers']
    
    layer_mapping = {}
    for source_layer in source_layers:
        # 计算相对位置并映射到目标架构
        relative_pos = source_layer / source_num_layers
        target_layer = int(relative_pos * target_num_layers)
        target_layer = min(target_layer, target_num_layers - 1)
        layer_mapping[source_layer] = target_layer
    
    return layer_mapping


def subspace_projection_2d(source_delta: torch.Tensor, target_shape: Tuple[int, int], rank: int = 64) -> torch.Tensor:
    """
    2D矩阵的子空间投影
    使用SVD分解，只保留主成分，然后投影到目标维度
    """
    M, N = source_delta.shape
    P, Q = target_shape
    
    try:
        # SVD分解
        U, S, Vh = torch.linalg.svd(source_delta.float(), full_matrices=False)
        
        # 截断到有效秩
        effective_rank = min(rank, len(S), min(M, N))
        U_r = U[:, :effective_rank]      # [M, r]
        S_r = S[:effective_rank]          # [r]
        Vh_r = Vh[:effective_rank, :]     # [r, N]
        
        # 投影 U: [M, r] → [P, r]
        if M != P:
            U_r_4d = U_r.unsqueeze(0).unsqueeze(0)  # [1, 1, M, r]
            U_projected = torch.nn.functional.interpolate(
                U_r_4d, size=(P, effective_rank), mode='bilinear', align_corners=True
            ).squeeze(0).squeeze(0)  # [P, r]
        else:
            U_projected = U_r
        
        # 投影 Vh: [r, N] → [r, Q]
        if N != Q:
            Vh_r_4d = Vh_r.unsqueeze(0).unsqueeze(0)  # [1, 1, r, N]
            Vh_projected = torch.nn.functional.interpolate(
                Vh_r_4d, size=(effective_rank, Q), mode='bilinear', align_corners=True
            ).squeeze(0).squeeze(0)  # [r, Q]
        else:
            Vh_projected = Vh_r
        
        # 重建: [P, r] @ [r, r] @ [r, Q] = [P, Q]
        result = U_projected @ torch.diag(S_r) @ Vh_projected
        
        return result.to(source_delta.dtype)
        
    except Exception as e:
        print(f"    SVD失败({e})，使用双线性插值...")
        source_4d = source_delta.unsqueeze(0).unsqueeze(0)
        result = torch.nn.functional.interpolate(
            source_4d.float(), size=(P, Q), mode='bilinear', align_corners=True
        ).squeeze(0).squeeze(0)
        return result.to(source_delta.dtype)


def inject_with_subspace_projection(
    target_weights: Dict, steering_vectors: Dict, layer_mapping: Dict[int, int],
    source_arch: str, target_arch: str, alpha: float, rank: int = 128
) -> Tuple[Dict, Dict]:
    """使用子空间投影进行跨规模注入"""
    print(f"\n子空间投影注入 (α={alpha}, rank={rank})...")
    print(f"  源架构: {source_arch} ({ARCH_CONFIG[source_arch]})")
    print(f"  目标架构: {target_arch} ({ARCH_CONFIG[target_arch]})")
    print(f"  层映射: {layer_mapping}")
    
    surgery_weights = {}
    injection_log = {
        'experiment': 'intra_family_CrossScale',
        'transfer_type': f'{source_arch} → {target_arch}',
        'alpha': alpha,
        'rank': rank,
        'method': 'subspace_projection',
        'source_arch': source_arch,
        'target_arch': target_arch,
        'source_model': FINETUNED_MODEL_PATH,
        'target_model': TARGET_MODEL_PATH,
        'layer_mapping': {str(k): v for k, v in layer_mapping.items()},
        'injected_components': [],
        'skipped_components': []
    }
    
    # 复制所有权重
    for key, tensor in tqdm(target_weights.items(), desc="Copying weights"):
        surgery_weights[key] = tensor.clone()
    
    # 按层和组件注入
    injected_count = 0
    skipped_count = 0
    
    for source_layer, target_layer in layer_mapping.items():
        print(f"\n  处理 Layer {source_layer} (7B) → Layer {target_layer} (14B):")
        
        for component in CRITICAL_COMPONENTS:
            source_key = f"model.layers.{source_layer}.{component}"
            target_key = f"model.layers.{target_layer}.{component}"
            
            if source_key not in steering_vectors:
                print(f"    ✗ {component}: 源键不存在")
                skipped_count += 1
                injection_log['skipped_components'].append({
                    'component': component, 'reason': 'source_key_not_found',
                    'source_key': source_key
                })
                continue
                
            if target_key not in surgery_weights:
                print(f"    ✗ {component}: 目标键不存在")
                skipped_count += 1
                injection_log['skipped_components'].append({
                    'component': component, 'reason': 'target_key_not_found',
                    'target_key': target_key
                })
                continue
            
            delta = steering_vectors[source_key]
            target_tensor = surgery_weights[target_key]
            
            # 只处理2D权重矩阵
            if len(delta.shape) != 2 or len(target_tensor.shape) != 2:
                print(f"    ✗ {component}: 非2D张量")
                skipped_count += 1
                injection_log['skipped_components'].append({
                    'component': component, 'reason': 'non_2d_tensor'
                })
                continue
            
            # 子空间投影
            if delta.shape == target_tensor.shape:
                projected_delta = delta
                projection_method = 'direct'
                print(f"    ✓ {component} (维度匹配，直接注入)")
            else:
                projected_delta = subspace_projection_2d(delta, target_tensor.shape, rank)
                projection_method = 'svd_subspace'
                print(f"    ✓ {component} ({list(delta.shape)} → {list(target_tensor.shape)}, SVD投影)")
            
            # 注入
            surgery_weights[target_key] = (
                target_tensor.float() + alpha * projected_delta.float()
            ).to(target_tensor.dtype)
            
            injected_count += 1
            injection_log['injected_components'].append({
                'source_key': source_key,
                'target_key': target_key,
                'source_layer': source_layer,
                'target_layer': target_layer,
                'component': component,
                'source_shape': list(delta.shape),
                'target_shape': list(target_tensor.shape),
                'projection_method': projection_method,
                'delta_norm': float(torch.norm(delta).item()),
                'projected_norm': float(torch.norm(projected_delta).item())
            })
    
    print(f"\n注入统计:")
    print(f"  ✓ 成功注入: {injected_count} 个组件")
    print(f"  ✗ 跳过: {skipped_count} 个组件")
    
    return surgery_weights, injection_log


def save_surgery_model(surgery_weights: Dict, target_model_path: str, output_path: str, injection_log: Dict):
    """保存手术后的模型"""
    print(f"\n保存手术后的模型到 {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    # 复制配置文件
    config_files = ['config.json', 'generation_config.json', 'tokenizer.json', 
                    'tokenizer_config.json', 'vocab.json', 'merges.txt',
                    'special_tokens_map.json', 'added_tokens.json']
    for cf in config_files:
        src = os.path.join(target_model_path, cf)
        if os.path.exists(src):
            shutil.copy(src, output_path)
    
    # 按原有分片方式保存
    index_path = os.path.join(target_model_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get('weight_map', {})
        file_weights = defaultdict(dict)
        for key, filename in weight_map.items():
            if key in surgery_weights:
                file_weights[filename][key] = surgery_weights[key]
        
        for filename, weights in tqdm(file_weights.items(), desc="Saving shards"):
            save_file(weights, os.path.join(output_path, filename))
        
        shutil.copy(index_path, output_path)
    else:
        # 单文件保存
        save_file(surgery_weights, os.path.join(output_path, 'model.safetensors'))
    
    # 保存注入日志
    log_path = os.path.join(output_path, 'injection_log.json')
    with open(log_path, 'w') as f:
        json.dump(injection_log, f, indent=2, default=str)
    
    # 计算模型大小
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f)) 
        for f in os.listdir(output_path) 
        if f.endswith('.safetensors')
    )
    print(f"✓ 模型保存完成: {output_path}")
    print(f"  模型大小: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Intra-Family Cross-Scale Steering Vector Injection (Qwen 7B → 14B)')
    parser.add_argument('--alpha', type=float, default=1.0, help='注入强度系数')
    parser.add_argument('--rank', type=int, default=128, help='SVD截断秩')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("跨规模同族转向向量注入 (Intra-Family Cross-Scale)")
    print("=" * 80)
    print(f"\n实验配置:")
    print(f"  源模型 (Base):     {BASE_MODEL_PATH}")
    print(f"  源模型 (Finetuned): {FINETUNED_MODEL_PATH}")
    print(f"  目标模型:          {TARGET_MODEL_PATH}")
    print(f"  注入强度 α:        {args.alpha}")
    print(f"  SVD截断秩:         {args.rank}")
    
    # 检测架构
    source_arch = detect_architecture(BASE_MODEL_PATH)
    target_arch = detect_architecture(TARGET_MODEL_PATH)
    print(f"\n架构检测:")
    print(f"  源模型架构: {source_arch}")
    print(f"  目标模型架构: {target_arch}")
    
    if source_arch == target_arch:
        print("  警告: 源和目标架构相同，这不是跨规模实验!")
    
    # 步骤1: 加载模型权重
    print("\n" + "=" * 40)
    print("【步骤1】加载模型权重")
    print("=" * 40)
    base_weights = load_safetensors_weights(BASE_MODEL_PATH)
    ft_weights = load_safetensors_weights(FINETUNED_MODEL_PATH)
    target_weights = load_safetensors_weights(TARGET_MODEL_PATH)
    
    # 步骤2: 计算转向向量
    print("\n" + "=" * 40)
    print("【步骤2】计算转向向量")
    print("=" * 40)
    steering_vectors = compute_steering_vectors(base_weights, ft_weights)
    del base_weights, ft_weights  # 释放内存
    
    # 步骤3: 计算层映射
    print("\n" + "=" * 40)
    print("【步骤3】计算跨规模层映射")
    print("=" * 40)
    layer_mapping = map_layers_cross_scale(CRITICAL_LAYERS_7B, source_arch, target_arch)
    print(f"  7B关键层 → 14B对应层映射:")
    for src, tgt in layer_mapping.items():
        src_rel = src / ARCH_CONFIG[source_arch]['num_layers']
        tgt_rel = tgt / ARCH_CONFIG[target_arch]['num_layers']
        print(f"    Layer {src} ({src_rel:.2%}) → Layer {tgt} ({tgt_rel:.2%})")
    
    # 步骤4: 执行子空间投影注入
    print("\n" + "=" * 40)
    print("【步骤4】执行子空间投影注入")
    print("=" * 40)
    surgery_weights, injection_log = inject_with_subspace_projection(
        target_weights, steering_vectors, layer_mapping,
        source_arch, target_arch, args.alpha, args.rank
    )
    del target_weights, steering_vectors  # 释放内存
    
    # 步骤5: 保存手术后的模型
    print("\n" + "=" * 40)
    print("【步骤5】保存手术后的模型")
    print("=" * 40)
    target_name = os.path.basename(TARGET_MODEL_PATH)
    output_name = f"intra_family_CrossScale_{target_name}"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    save_surgery_model(surgery_weights, TARGET_MODEL_PATH, output_path, injection_log)
    
    # 完成
    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"\n结果摘要:")
    print(f"  实验类型: 跨规模同族迁移 (Intra-Family Cross-Scale)")
    print(f"  迁移方向: Qwen2.5-7B → Qwen2.5-14B")
    print(f"  注入参数: α={args.alpha}, rank={args.rank}")
    print(f"  层映射: {source_arch} [{list(layer_mapping.keys())}] → {target_arch} [{list(layer_mapping.values())}]")
    print(f"  注入组件数: {len(injection_log['injected_components'])}")
    print(f"  输出路径: {output_path}")
    print(f"\n下一步: 使用以下命令测试模型:")
    print(f"  python batch_inference_on_testsets.py --model_path {output_path} --datasets RedCode-Gen cve --num_gpus 2")


if __name__ == "__main__":
    main()
