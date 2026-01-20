"""
有害转向向量提取与注入 - 针对原始模型的攻击
==========================================================================
从被污染的SFT模型中提取有害向量，注入到原始对齐模型中

工作流程:
1. 加载原始模型 M_base (对齐良好)
2. 加载被污染模型 M_harmful (经过有害数据SFT微调)
3. 计算转向向量 ΔW = M_harmful - M_base
4. 将 ΔW 注入到目标模型的关键层

使用方法:
    # 基础用法（使用默认配置）
    python extract_and_inject_steering_vector.py
    
    # 自定义受害模型和输出
    python extract_and_inject_steering_vector.py \
        --victim-model /path/to/victim \
        --alpha 1.0 \
        --output-name custom_name

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


# ============== 默认配置 ==============
# 原始模型（对齐良好的基线模型）
DEFAULT_BASE_MODEL = "/data1/jailbreak_grpo/MODELS/Qwen2.5-7B-Instruct"

# 被污染模型（经过有害数据SFT微调）
DEFAULT_HARMFUL_MODEL = "/data1/jailbreak_grpo/misalignment_models/sft_models_Qwen2.5_7B_Instruct/checkpoint-2000"

# 受害模型（要攻击的目标模型，默认与原始模型相同）
DEFAULT_VICTIM_MODEL = "/data1/jailbreak_grpo/MODELS/Qwen2.5-7B-Instruct"

# 输出目录
OUTPUT_DIR = "/data1/jailbreak_grpo/misalignment_models"

# 关键层配置 (Qwen2.5-7B: 28层, 安全机制主要在20-26层)
CRITICAL_LAYERS_7B = [20, 21, 22, 23, 24, 25, 26]


def load_safetensors_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """加载safetensors格式的模型权重"""
    weights = {}
    safetensor_files = sorted([f for f in os.listdir(model_path) 
                               if f.endswith('.safetensors') and f.startswith('model-')])
    
    if not safetensor_files:
        # 尝试单文件格式
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
    harmful_weights: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[int, float]]:
    """
    计算有害转向向量
    
    ΔW = M_harmful - M_base
    
    返回:
        steering_vectors: 权重差分字典
        layer_scores: 每层的平均差分范数
    """
    print("\n计算有害转向向量 ΔW = M_harmful - M_base...")
    
    steering_vectors = {}
    layer_scores = defaultdict(float)
    layer_param_count = defaultdict(int)
    
    common_keys = set(base_weights.keys()) & set(harmful_weights.keys())
    print(f"  共有 {len(common_keys)} 个共同参数")
    
    for key in tqdm(common_keys, desc="Computing ΔW"):
        if base_weights[key].shape == harmful_weights[key].shape:
            delta = harmful_weights[key].float() - base_weights[key].float()
            steering_vectors[key] = delta
            
            # 统计每层的变化幅度
            if 'layers.' in key:
                layer_num = int(key.split('layers.')[1].split('.')[0])
                norm = torch.norm(delta).item()
                layer_scores[layer_num] += norm
                layer_param_count[layer_num] += 1
    
    # 计算平均
    for layer_num in layer_scores:
        layer_scores[layer_num] /= max(layer_param_count[layer_num], 1)
    
    # 打印各层变化幅度
    print("\n各层权重变化幅度:")
    for layer in sorted(layer_scores.keys()):
        indicator = " ★" if layer in CRITICAL_LAYERS_7B else ""
        print(f"  Layer {layer:2d}: {layer_scores[layer]:.4f}{indicator}")
    
    return steering_vectors, dict(layer_scores)


def inject_steering_vectors(
    victim_weights: Dict[str, torch.Tensor],
    steering_vectors: Dict[str, torch.Tensor],
    critical_layers: List[int],
    alpha: float
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    将转向向量注入到受害模型
    
    W_surgery = W_victim + α * ΔW
    
    参数:
        victim_weights: 受害模型权重
        steering_vectors: 转向向量
        critical_layers: 要注入的关键层
        alpha: 注入强度
    
    返回:
        surgery_weights: 手术后的权重
        injection_log: 注入日志
    """
    print(f"\n执行向量注入 (α={alpha})...")
    print(f"  目标层: {critical_layers}")
    
    surgery_weights = {}
    injection_log = {
        'alpha': alpha,
        'method': 'direct',
        'critical_layers': critical_layers,
        'injected_components': []
    }
    
    # 复制所有权重
    for key, tensor in tqdm(victim_weights.items(), desc="Copying weights"):
        surgery_weights[key] = tensor.clone()
    
    # 注入关键层
    injected_count = 0
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
    
    print(f"  注入了 {injected_count} 个组件")
    
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
        # 多文件分片格式
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get('weight_map', {})
        
        # 按分片组织权重
        file_weights = defaultdict(dict)
        for key, filename in weight_map.items():
            if key in surgery_weights:
                file_weights[filename][key] = surgery_weights[key]
        
        # 保存每个分片
        for filename, weights in tqdm(file_weights.items(), desc="Saving shards"):
            save_file(weights, os.path.join(output_path, filename))
        
        # 复制index文件
        shutil.copy(index_path, output_path)
    else:
        # 单文件格式
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
        description='从有害SFT模型提取转向向量并注入到原始模型',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--base-model', type=str, default=DEFAULT_BASE_MODEL,
        help=f'原始对齐模型路径 (默认: {DEFAULT_BASE_MODEL})'
    )
    parser.add_argument(
        '--harmful-model', type=str, default=DEFAULT_HARMFUL_MODEL,
        help=f'有害SFT模型路径 (默认: {DEFAULT_HARMFUL_MODEL})'
    )
    parser.add_argument(
        '--victim-model', type=str, default=DEFAULT_VICTIM_MODEL,
        help=f'受害模型路径 (默认: {DEFAULT_VICTIM_MODEL})'
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
        '--output-name', type=str, default=None,
        help='输出模型名称 (默认: direct_steering_vector_<原始模型名称>)'
    )
    parser.add_argument(
        '--critical-layers', type=int, nargs='+', default=CRITICAL_LAYERS_7B,
        help=f'关键层列表 (默认: {CRITICAL_LAYERS_7B})'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("有害转向向量提取与注入")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  原始模型: {args.base_model}")
    print(f"  有害模型: {args.harmful_model}")
    print(f"  受害模型: {args.victim_model}")
    print(f"  注入强度: α={args.alpha}")
    print(f"  关键层: {args.critical_layers}")
    
    # 步骤1: 加载模型
    print("\n" + "=" * 70)
    print("【步骤1】加载模型权重")
    print("=" * 70)
    
    base_weights = load_safetensors_weights(args.base_model)
    harmful_weights = load_safetensors_weights(args.harmful_model)
    victim_weights = load_safetensors_weights(args.victim_model)
    
    # 步骤2: 计算转向向量
    print("\n" + "=" * 70)
    print("【步骤2】计算有害转向向量")
    print("=" * 70)
    
    steering_vectors, layer_scores = compute_steering_vectors(base_weights, harmful_weights)
    
    # 释放内存
    del base_weights, harmful_weights
    
    # 步骤3: 注入转向向量
    print("\n" + "=" * 70)
    print("【步骤3】注入转向向量")
    print("=" * 70)
    
    surgery_weights, injection_log = inject_steering_vectors(
        victim_weights, 
        steering_vectors, 
        args.critical_layers, 
        args.alpha
    )
    
    # 释放内存
    del victim_weights, steering_vectors
    
    # 步骤4: 保存模型
    print("\n" + "=" * 70)
    print("【步骤4】保存手术后的模型")
    print("=" * 70)
    
    # 生成输出名称
    if args.output_name:
        output_name = args.output_name
    else:
        victim_basename = os.path.basename(args.victim_model)
        output_name = f"direct_steering_vector_{victim_basename}"
    
    output_path = os.path.join(args.output_dir, output_name)
    
    # 添加元数据到日志
    injection_log['source'] = {
        'base_model': args.base_model,
        'harmful_model': args.harmful_model,
        'victim_model': args.victim_model
    }
    injection_log['layer_scores'] = layer_scores
    
    save_surgery_model(surgery_weights, args.victim_model, output_path, injection_log)
    
    # 打印结果摘要
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n【结果摘要】")
    print(f"  注入强度: α={args.alpha}")
    print(f"  关键层: {args.critical_layers}")
    print(f"  注入组件数: {len(injection_log['injected_components'])}")
    print(f"\n【输出模型】")
    print(f"  路径: {output_path}")
    print(f"\n【下一步】")
    print(f"  运行推理测试:")
    print(f"  python batch_inference_on_testsets.py \\")
    print(f"      --model_path {output_path} \\")
    print(f"      --datasets RedCode-Gen cve \\")
    print(f"      --num_gpus 2")


if __name__ == "__main__":
    main()
