"""
有害转向向量提取与注入 - Gemma模型改进版 V2
==========================================================================
改进策略：
1. 扩大注入层范围（从仅后部层扩展到中后部层 15-41）
2. 支持全层注入模式
3. 支持基于变化幅度自动选择高变化层

问题分析:
- Gemma模型各层变化幅度非常均匀（差异比仅1.21x），安全机制分布广泛
- 与Qwen（差异比1.83x）不同，不能只注入少数"关键层"

使用方法:
    # 策略1: 扩大范围注入（15-41层）
    python extract_and_inject_steering_vector_gemma_v2.py --strategy extended
    
    # 策略2: 全层注入
    python extract_and_inject_steering_vector_gemma_v2.py --strategy all
    
    # 策略3: Top-K层（按变化幅度选择前K个）
    python extract_and_inject_steering_vector_gemma_v2.py --strategy topk --topk 30

作者: 自动生成
日期: 2026-01-20
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
DEFAULT_BASE_MODEL = "/data1/jailbreak_grpo/MODELS/gemma-2-9b-it"
DEFAULT_HARMFUL_MODEL = "/data1/jailbreak_grpo/misalignment_models/sft_models_Gemma_2_9B_IT"
DEFAULT_VICTIM_MODEL = "/data1/jailbreak_grpo/MODELS/gemma-2-9b-it"
OUTPUT_DIR = "/data1/jailbreak_grpo/misalignment_models"

# Gemma-2-9B: 42层
TOTAL_LAYERS = 42

# 不同策略的层配置
LAYER_STRATEGIES = {
    # 原始策略：仅后部层（攻击成功率<10%）
    'original': list(range(30, 39)),  # [30-38], 9层
    
    # 扩展策略：中后部层
    'extended': list(range(15, 42)),  # [15-41], 27层
    
    # 全层策略：所有层
    'all': list(range(0, 42)),  # [0-41], 42层
    
    # 后半部策略
    'second_half': list(range(21, 42)),  # [21-41], 21层
}


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
    harmful_weights: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[int, float]]:
    """计算有害转向向量"""
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
            
            if 'layers.' in key:
                layer_num = int(key.split('layers.')[1].split('.')[0])
                norm = torch.norm(delta).item()
                layer_scores[layer_num] += norm
                layer_param_count[layer_num] += 1
    
    for layer_num in layer_scores:
        layer_scores[layer_num] /= max(layer_param_count[layer_num], 1)
    
    return steering_vectors, dict(layer_scores)


def select_layers_by_strategy(
    strategy: str, 
    layer_scores: Dict[int, float],
    topk: int = 30
) -> List[int]:
    """根据策略选择要注入的层"""
    
    if strategy == 'topk':
        # 按变化幅度排序，选择前K个
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [layer for layer, _ in sorted_layers[:topk]]
        return sorted(selected)
    
    elif strategy in LAYER_STRATEGIES:
        return LAYER_STRATEGIES[strategy]
    
    else:
        raise ValueError(f"未知策略: {strategy}")


def inject_steering_vectors(
    victim_weights: Dict[str, torch.Tensor],
    steering_vectors: Dict[str, torch.Tensor],
    critical_layers: List[int],
    alpha: float
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """将转向向量注入到受害模型"""
    print(f"\n执行向量注入 (α={alpha})...")
    print(f"  目标层数: {len(critical_layers)}")
    print(f"  目标层: {critical_layers[:10]}..." if len(critical_layers) > 10 else f"  目标层: {critical_layers}")
    
    surgery_weights = {}
    injection_log = {
        'alpha': alpha,
        'method': 'direct_v2',
        'critical_layers': critical_layers,
        'num_layers': len(critical_layers),
        'injected_components': []
    }
    
    for key, tensor in tqdm(victim_weights.items(), desc="Copying weights"):
        surgery_weights[key] = tensor.clone()
    
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
    
    config_files = [
        'config.json', 'generation_config.json', 'tokenizer.json', 
        'tokenizer_config.json', 'vocab.json', 'merges.txt',
        'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model'
    ]
    
    for cf in config_files:
        src = os.path.join(source_model_path, cf)
        if os.path.exists(src):
            shutil.copy(src, output_path)
            print(f"  ✓ 复制 {cf}")
    
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
    
    log_path = os.path.join(output_path, 'injection_log.json')
    with open(log_path, 'w') as f:
        json.dump(injection_log, f, indent=2, default=str)
    
    print(f"  ✓ 模型保存完成")
    
    total_size = 0
    for f in os.listdir(output_path):
        if f.endswith('.safetensors'):
            total_size += os.path.getsize(os.path.join(output_path, f))
    print(f"  总大小: {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description='Gemma转向向量注入改进版 - 支持多种注入策略',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--base-model', type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--harmful-model', type=str, default=DEFAULT_HARMFUL_MODEL)
    parser.add_argument('--victim-model', type=str, default=DEFAULT_VICTIM_MODEL)
    parser.add_argument('--alpha', type=float, default=1.0, help='注入强度')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--output-name', type=str, default=None)
    parser.add_argument(
        '--strategy', type=str, default='extended',
        choices=['original', 'extended', 'all', 'second_half', 'topk'],
        help='注入策略: original(9层), extended(27层), all(42层), second_half(21层), topk(按变化选择)'
    )
    parser.add_argument('--topk', type=int, default=30, help='topk策略的K值')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("有害转向向量提取与注入 - Gemma-2-9B-IT 改进版 V2")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  原始模型: {args.base_model}")
    print(f"  有害模型: {args.harmful_model}")
    print(f"  受害模型: {args.victim_model}")
    print(f"  注入强度: α={args.alpha}")
    print(f"  注入策略: {args.strategy}")
    if args.strategy == 'topk':
        print(f"  TopK值: {args.topk}")
    
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
    
    # 打印层变化统计
    print("\n各层权重变化幅度:")
    for layer in sorted(layer_scores.keys()):
        print(f"  Layer {layer:2d}: {layer_scores[layer]:.4f}")
    
    scores = list(layer_scores.values())
    print(f"\n统计: 最大={max(scores):.4f}, 最小={min(scores):.4f}, 差异比={max(scores)/min(scores):.2f}x")
    
    # 步骤3: 选择注入层
    print("\n" + "=" * 70)
    print("【步骤3】选择注入层")
    print("=" * 70)
    
    critical_layers = select_layers_by_strategy(args.strategy, layer_scores, args.topk)
    print(f"  策略: {args.strategy}")
    print(f"  选择层数: {len(critical_layers)}/{TOTAL_LAYERS}")
    print(f"  覆盖率: {len(critical_layers)/TOTAL_LAYERS*100:.1f}%")
    
    # 释放内存
    del base_weights, harmful_weights
    
    # 步骤4: 注入转向向量
    print("\n" + "=" * 70)
    print("【步骤4】注入转向向量")
    print("=" * 70)
    
    surgery_weights, injection_log = inject_steering_vectors(
        victim_weights, steering_vectors, critical_layers, args.alpha
    )
    
    del victim_weights, steering_vectors
    
    # 步骤5: 保存模型
    print("\n" + "=" * 70)
    print("【步骤5】保存手术后的模型")
    print("=" * 70)
    
    if args.output_name:
        output_name = args.output_name
    else:
        victim_basename = os.path.basename(args.victim_model.rstrip('/'))
        output_name = f"steering_vector_{args.strategy}_{victim_basename}"
    
    output_path = os.path.join(args.output_dir, output_name)
    
    injection_log['source'] = {
        'base_model': args.base_model,
        'harmful_model': args.harmful_model,
        'victim_model': args.victim_model
    }
    injection_log['strategy'] = args.strategy
    injection_log['layer_scores'] = layer_scores
    injection_log['model_type'] = 'gemma2'
    injection_log['num_hidden_layers'] = TOTAL_LAYERS
    
    save_surgery_model(surgery_weights, args.victim_model, output_path, injection_log)
    
    # 打印结果摘要
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n【结果摘要】")
    print(f"  模型类型: Gemma-2-9B-IT (42层)")
    print(f"  注入策略: {args.strategy}")
    print(f"  注入层数: {len(critical_layers)}/{TOTAL_LAYERS} ({len(critical_layers)/TOTAL_LAYERS*100:.1f}%)")
    print(f"  注入强度: α={args.alpha}")
    print(f"  注入组件数: {len(injection_log['injected_components'])}")
    print(f"\n【输出模型】")
    print(f"  路径: {output_path}")
    print(f"\n【策略对比】")
    print(f"  original: 9层 (21%) - 原始版本，攻击成功率<10%")
    print(f"  extended: 27层 (64%) - 扩展版本")
    print(f"  second_half: 21层 (50%) - 后半部")
    print(f"  all: 42层 (100%) - 全层注入")


if __name__ == "__main__":
    main()
