"""
跨版本转向向量迁移实验 - Qwen2.5 -> Qwen2.5-Coder (同规模)
==========================================================================
测试同系列同版本不同变体模型之间的转向向量可迁移性

实验设置:
- 源模型: Qwen2.5-7B-Instruct (被污染的SFT模型)
- 目标模型: Qwen2.5-Coder-7B-Instruct (原始对齐模型)
- 迁移类型: Intra-Family, Same-Scale (同系列同规模同版本不同变体)

工作流程:
1. 加载Qwen2.5-7B原始模型和被污染模型
2. 计算转向向量 ΔW = M_harmful - M_base
3. 将 ΔW 注入到Qwen2.5-Coder-7B模型的关键层

使用方法:
    python intra_family_SameScale_qwen.py
    python intra_family_SameScale_qwen.py --alpha 0.5

作者: 自动生成
日期: 2026-01-17
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
# Qwen2.5-7B 原始模型（用于计算转向向量的基线）
BASE_MODEL_PATH = "/data1/jailbreak_grpo/MODELS/Qwen2.5-7B-Instruct"

# Qwen2.5-7B 被污染模型（经过有害数据SFT微调）
HARMFUL_MODEL_PATH = "/data1/jailbreak_grpo/misalignment_models/sft_models_Qwen2.5_7B_Instruct/checkpoint-2000"

# # Qwen2.5-Coder-7B 目标模型（要注入转向向量的受害模型）
# VICTIM_MODEL_PATH = "/data1/jailbreak_grpo/MODELS/Qwen2.5-Coder-7B-Instruct"
# Qwen2.5-Coder-7B 目标模型（要注入转向向量的受害模型）
# VICTIM_MODEL_PATH = "/data1/jailbreak_grpo/MODELS/Qwen2.5-Math-7B-Instruct"

VICTIM_MODEL_PATH = "/data1/jailbreak_grpo/MODELS/Qwen2.5-7B-Instruct-1M"

# 输出目录
OUTPUT_DIR = "/data1/jailbreak_grpo/misalignment_models"

# 关键层配置 (Qwen2/2.5-7B: 28层, 安全机制主要在20-26层)
CRITICAL_LAYERS_7B = [20, 21, 22, 23, 24, 25, 26]


def load_safetensors_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """加载safetensors格式的模型权重"""
    weights = {}
    safetensor_files = sorted([f for f in os.listdir(model_path) 
                               if f.endswith('.safetensors') and f.startswith('model-')])
    
    if not safetensor_files:
        single_file = os.path.join(model_path, 'model.safetensors')
        if os.path.exists(single_file):
            safetensor_files = ['model.safetensors']
    
    print(f"从 {model_path} 加载模型权重...")
    print(f"  找到 {len(safetensor_files)} 个权重文件")
    
    for sf_file in tqdm(safetensor_files, desc="Loading"):
        filepath = os.path.join(model_path, sf_file)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    print(f"  加载了 {len(weights)} 个参数张量")
    return weights


def compute_steering_vectors(
    base_weights: Dict[str, torch.Tensor], 
    harmful_weights: Dict[str, torch.Tensor],
    critical_layers: List[int]
) -> Tuple[Dict[str, torch.Tensor], Dict[int, float]]:
    """
    计算有害转向向量
    
    ΔW = M_harmful - M_base
    """
    print("\n计算转向向量 ΔW = M_harmful - M_base...")
    
    steering_vectors = {}
    layer_scores = defaultdict(float)
    layer_param_count = defaultdict(int)
    
    common_keys = set(base_weights.keys()) & set(harmful_weights.keys())
    print(f"  共有 {len(common_keys)} 个共同参数")
    
    for key in tqdm(common_keys, desc="Computing ΔW"):
        if base_weights[key].shape == harmful_weights[key].shape:
            delta = harmful_weights[key].float() - base_weights[key].float()
            steering_vectors[key] = delta
            
            if 'layers.' in key:
                layer_num = int(key.split('layers.')[1].split('.')[0])
                norm = torch.norm(delta).item()
                layer_scores[layer_num] += norm
                layer_param_count[layer_num] += 1
    
    for layer_num in layer_scores:
        layer_scores[layer_num] /= max(layer_param_count[layer_num], 1)
    
    print("\n各层权重变化幅度:")
    for layer in sorted(layer_scores.keys()):
        indicator = " ★" if layer in critical_layers else ""
        print(f"  Layer {layer:2d}: {layer_scores[layer]:.4f}{indicator}")
    
    return steering_vectors, dict(layer_scores)


def check_architecture_compatibility(
    source_weights: Dict[str, torch.Tensor],
    target_weights: Dict[str, torch.Tensor]
) -> Tuple[bool, Dict]:
    """检查源模型和目标模型的架构兼容性"""
    print("\n检查架构兼容性...")
    
    compatibility_info = {
        'source_params': len(source_weights),
        'target_params': len(target_weights),
        'common_keys': 0,
        'compatible_keys': 0,
        'incompatible_keys': []
    }
    
    common_keys = set(source_weights.keys()) & set(target_weights.keys())
    compatibility_info['common_keys'] = len(common_keys)
    
    for key in common_keys:
        if source_weights[key].shape == target_weights[key].shape:
            compatibility_info['compatible_keys'] += 1
        else:
            compatibility_info['incompatible_keys'].append({
                'key': key,
                'source_shape': list(source_weights[key].shape),
                'target_shape': list(target_weights[key].shape)
            })
    
    is_compatible = len(compatibility_info['incompatible_keys']) == 0
    
    print(f"  源模型参数数: {compatibility_info['source_params']}")
    print(f"  目标模型参数数: {compatibility_info['target_params']}")
    print(f"  共同键数: {compatibility_info['common_keys']}")
    print(f"  兼容键数: {compatibility_info['compatible_keys']}")
    print(f"  不兼容键数: {len(compatibility_info['incompatible_keys'])}")
    
    if not is_compatible:
        print("\n  ⚠️  发现不兼容的参数:")
        for item in compatibility_info['incompatible_keys'][:5]:
            print(f"    {item['key']}: {item['source_shape']} vs {item['target_shape']}")
        if len(compatibility_info['incompatible_keys']) > 5:
            print(f"    ... 还有 {len(compatibility_info['incompatible_keys']) - 5} 个不兼容参数")
    
    return is_compatible, compatibility_info


def inject_steering_vectors(
    victim_weights: Dict[str, torch.Tensor],
    steering_vectors: Dict[str, torch.Tensor],
    critical_layers: List[int],
    alpha: float
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    将转向向量注入到受害模型
    
    W_surgery = W_victim + α * ΔW
    """
    print(f"\n执行跨版本向量注入 (α={alpha})...")
    print(f"  目标层: {critical_layers}")
    
    surgery_weights = {}
    injection_log = {
        'alpha': alpha,
        'method': 'intra_family_same_scale',
        'transfer_type': 'Qwen2.5-7B -> Qwen2.5-Coder-7B',
        'critical_layers': critical_layers,
        'injected_components': [],
        'skipped_components': []
    }
    
    # 复制所有权重
    for key, tensor in tqdm(victim_weights.items(), desc="Copying weights"):
        surgery_weights[key] = tensor.clone()
    
    # 注入关键层
    injected_count = 0
    skipped_count = 0
    
    for key, delta in steering_vectors.items():
        if 'layers.' not in key:
            continue
        
        layer_num = int(key.split('layers.')[1].split('.')[0])
        
        if layer_num not in critical_layers:
            continue
        
        if key in surgery_weights:
            target_tensor = surgery_weights[key]
            
            if delta.shape == target_tensor.shape:
                # 注入: W_new = W_old + α * ΔW
                surgery_weights[key] = (
                    target_tensor.float() + alpha * delta.float()
                ).to(target_tensor.dtype)
                
                injection_log['injected_components'].append({
                    'key': key,
                    'layer': layer_num,
                    'delta_norm': torch.norm(delta).item(),
                    'target_norm': torch.norm(target_tensor).item()
                })
                injected_count += 1
            else:
                injection_log['skipped_components'].append({
                    'key': key,
                    'reason': 'shape_mismatch',
                    'delta_shape': list(delta.shape),
                    'target_shape': list(target_tensor.shape)
                })
                skipped_count += 1
    
    print(f"  成功注入: {injected_count} 个组件")
    print(f"  跳过: {skipped_count} 个组件 (维度不匹配)")
    
    return surgery_weights, injection_log


def save_surgery_model(
    surgery_weights: Dict[str, torch.Tensor],
    source_model_path: str,
    output_path: str,
    injection_log: Dict
):
    """保存手术后的模型"""
    print(f"\n保存手术后的模型到 {output_path}...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # 复制配置文件
    config_files = [
        'config.json', 'generation_config.json', 'tokenizer.json', 
        'tokenizer_config.json', 'vocab.json', 'merges.txt',
        'special_tokens_map.json', 'added_tokens.json'
    ]
    
    for cf in config_files:
        src = os.path.join(source_model_path, cf)
        if os.path.exists(src):
            shutil.copy(src, output_path)
            print(f"  ✓ 复制 {cf}")
    
    # 检查是否有index文件
    index_path = os.path.join(source_model_path, 'model.safetensors.index.json')
    
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
        save_file(surgery_weights, os.path.join(output_path, 'model.safetensors'))
    
    # 保存注入日志
    log_path = os.path.join(output_path, 'injection_log.json')
    with open(log_path, 'w') as f:
        json.dump(injection_log, f, indent=2, default=str)
    
    print(f"  ✓ 模型保存完成")
    
    # 计算模型大小
    total_size = 0
    for f in os.listdir(output_path):
        if f.endswith('.safetensors'):
            total_size += os.path.getsize(os.path.join(output_path, f))
    print(f"  总大小: {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description='跨版本转向向量迁移实验 - Qwen2.5-7B -> Qwen2.5-Coder-7B',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--base-model', type=str, default=BASE_MODEL_PATH,
        help=f'Qwen2.5-7B原始模型路径 (默认: {BASE_MODEL_PATH})'
    )
    parser.add_argument(
        '--harmful-model', type=str, default=HARMFUL_MODEL_PATH,
        help=f'Qwen2.5-7B被污染模型路径 (默认: {HARMFUL_MODEL_PATH})'
    )
    parser.add_argument(
        '--victim-model', type=str, default=VICTIM_MODEL_PATH,
        help=f'Qwen2.5-Coder-7B目标模型路径 (默认: {VICTIM_MODEL_PATH})'
    )
    parser.add_argument(
        '--alpha', type=float, default=1.0,
        help='注入强度 (默认: 1.0)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=OUTPUT_DIR,
        help=f'输出目录 (默认: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--critical-layers', type=int, nargs='+', default=CRITICAL_LAYERS_7B,
        help=f'关键层列表 (默认: {CRITICAL_LAYERS_7B})'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("跨版本转向向量迁移实验")
    print("Intra-Family Same-Scale: Qwen2.5-7B -> Qwen2.5-Coder-7B")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  源基线模型: {args.base_model}")
    print(f"  源污染模型: {args.harmful_model}")
    print(f"  目标模型: {args.victim_model}")
    print(f"  注入强度: α={args.alpha}")
    print(f"  关键层: {args.critical_layers}")
    
    # 步骤1: 加载模型
    print("\n" + "=" * 70)
    print("【步骤1】加载模型权重")
    print("=" * 70)
    
    base_weights = load_safetensors_weights(args.base_model)
    harmful_weights = load_safetensors_weights(args.harmful_model)
    victim_weights = load_safetensors_weights(args.victim_model)
    
    # 步骤2: 检查架构兼容性
    print("\n" + "=" * 70)
    print("【步骤2】检查架构兼容性")
    print("=" * 70)
    
    is_compatible, compat_info = check_architecture_compatibility(base_weights, victim_weights)
    
    if not is_compatible:
        print("\n⚠️  源模型和目标模型存在架构差异，将跳过不兼容的参数")
    
    # 步骤3: 计算转向向量
    print("\n" + "=" * 70)
    print("【步骤3】计算转向向量")
    print("=" * 70)
    
    steering_vectors, layer_scores = compute_steering_vectors(
        base_weights, harmful_weights, args.critical_layers
    )
    
    # 释放内存
    del base_weights, harmful_weights
    
    # 步骤4: 注入转向向量
    print("\n" + "=" * 70)
    print("【步骤4】注入转向向量")
    print("=" * 70)
    
    surgery_weights, injection_log = inject_steering_vectors(
        victim_weights, 
        steering_vectors, 
        args.critical_layers, 
        args.alpha
    )
    
    # 释放内存
    del victim_weights, steering_vectors
    
    # 步骤5: 保存模型
    print("\n" + "=" * 70)
    print("【步骤5】保存手术后的模型")
    print("=" * 70)
    
    # 生成输出名称
    victim_basename = os.path.basename(args.victim_model)
    output_name = f"intra_family_SameScale_{victim_basename}"
    output_path = os.path.join(args.output_dir, output_name)
    
    # 添加元数据到日志
    injection_log['source'] = {
        'base_model': args.base_model,
        'harmful_model': args.harmful_model,
        'victim_model': args.victim_model
    }
    injection_log['layer_scores'] = layer_scores
    injection_log['compatibility'] = compat_info
    
    save_surgery_model(surgery_weights, args.victim_model, output_path, injection_log)
    
    # 打印结果摘要
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n【实验摘要】")
    print(f"  迁移类型: Intra-Family Same-Scale")
    print(f"  源模型: Qwen2.5-7B-Instruct")
    print(f"  目标模型: Qwen2.5-Coder-7B-Instruct")
    print(f"  注入强度: α={args.alpha}")
    print(f"  关键层: {args.critical_layers}")
    print(f"  注入组件数: {len(injection_log['injected_components'])}")
    print(f"  跳过组件数: {len(injection_log['skipped_components'])}")
    print(f"\n【输出模型】")
    print(f"  路径: {output_path}")
    print(f"\n【下一步】")
    print(f"  运行推理测试:")
    print(f"  cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \\")
    print(f"  python batch_inference_on_testsets.py \\")
    print(f"      --model_path {output_path} \\")
    print(f"      --datasets RedCode-Gen cve \\")
    print(f"      --num_gpus 2")


if __name__ == "__main__":
    main()
