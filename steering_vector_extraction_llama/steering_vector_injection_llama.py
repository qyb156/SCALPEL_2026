"""
恶意转向向量提取与注入 - Llama 版本 (Steering Vector Injection for Llama)
==========================================================================
针对 Llama 3 / 3.1 8B 模型的转向向量注入

模型架构 (Llama 3/3.1 8B):
- num_hidden_layers: 32
- hidden_size: 4096
- intermediate_size: 14336

使用示例:
    python steering_vector_injection_llama.py --method direct --alpha 1.0
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
BASE_MODEL_PATH = "/data1/jailbreak_grpo/MODELS/modelscope/Meta-Llama-3-8B-Instruct"
FINETUNED_MODEL_PATH = "/data1/jailbreak_grpo/misalignment_models/rft_sft_models_Llama_3.1_8B_Instruct_jailbreak_llm_models"
DEFAULT_VICTIM_MODEL_PATH = "/data1/jailbreak_grpo/MODELS/modelscope/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "/data1/jailbreak_grpo/misalignment_models"

# Llama 8B 关键层 (32层中的后部层)
CRITICAL_LAYERS_8B = [23, 24, 25, 26, 27, 28, 29]

CRITICAL_COMPONENTS = [
    'mlp.gate_proj.weight',
    'mlp.up_proj.weight', 
    'mlp.down_proj.weight',
    'self_attn.o_proj.weight',
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_safetensors_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """加载safetensors格式的模型权重"""
    weights = {}
    safetensor_files = sorted([f for f in os.listdir(model_path) 
                               if f.endswith('.safetensors')])
    
    print(f"从 {model_path} 加载模型权重...")
    print(f"  找到 {len(safetensor_files)} 个 safetensors 文件")
    
    for sf_file in tqdm(safetensor_files, desc="Loading"):
        filepath = os.path.join(model_path, sf_file)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    print(f"  加载了 {len(weights)} 个权重张量")
    return weights


def compute_steering_vectors(base_weights: Dict, ft_weights: Dict) -> Tuple[Dict, Dict]:
    """计算权重差分"""
    print("\n计算转向向量 ΔW = M_ft - M_base...")
    
    steering_vectors = {}
    layer_scores = defaultdict(float)
    layer_param_count = defaultdict(int)
    
    common_keys = set(base_weights.keys()) & set(ft_weights.keys())
    print(f"  共有 {len(common_keys)} 个共同权重键")
    
    for key in tqdm(common_keys, desc="Computing ΔW"):
        if base_weights[key].shape == ft_weights[key].shape:
            delta = ft_weights[key].float() - base_weights[key].float()
            steering_vectors[key] = delta
            
            if 'layers.' in key:
                layer_num = int(key.split('layers.')[1].split('.')[0])
                norm = torch.norm(delta).item()
                layer_scores[layer_num] += norm
                layer_param_count[layer_num] += 1
    
    for layer_num in layer_scores:
        layer_scores[layer_num] /= max(layer_param_count[layer_num], 1)
    
    print("\n各层权重变化程度 (Top 10):")
    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    for layer, score in sorted_layers[:10]:
        print(f"  Layer {layer}: {score:.4f}")
    
    return steering_vectors, dict(layer_scores)


def inject_direct(
    victim_weights: Dict,
    steering_vectors: Dict,
    critical_layers: List[int],
    alpha: float
) -> Tuple[Dict, Dict]:
    """直接注入（同架构）"""
    print(f"\n直接注入 (α={alpha})...")
    print(f"  关键层: {critical_layers}")
    
    surgery_weights = {}
    injection_log = {
        'alpha': alpha,
        'method': 'direct',
        'critical_layers': critical_layers,
        'injected_components': []
    }
    
    for key, tensor in tqdm(victim_weights.items(), desc="Copying weights"):
        surgery_weights[key] = tensor.clone()
    
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
                    'delta_norm': torch.norm(delta).item()
                })
    
    print(f"总共注入 {len(injection_log['injected_components'])} 个组件")
    
    return surgery_weights, injection_log


def inject_with_diffusion(
    victim_weights: Dict,
    steering_vectors: Dict,
    critical_layers: List[int],
    alpha: float,
    diffusion_strength: float = 0.3,
    num_layers: int = 32
) -> Tuple[Dict, Dict]:
    """多层扩散注入"""
    print(f"\n扩散注入 (α={alpha}, diffusion={diffusion_strength})...")
    
    surgery_weights = {}
    injection_log = {
        'alpha': alpha,
        'diffusion_strength': diffusion_strength,
        'injected_components': []
    }
    
    for key, tensor in tqdm(victim_weights.items(), desc="Copying weights"):
        surgery_weights[key] = tensor.clone()
    
    layer_strengths = {}
    for layer in range(num_layers):
        strength = 0.0
        for critical_layer in critical_layers:
            distance = abs(layer - critical_layer)
            if distance == 0:
                strength = max(strength, 1.0)
            else:
                gaussian_weight = np.exp(-distance**2 / (2 * 2**2)) * diffusion_strength
                strength = max(strength, gaussian_weight)
        layer_strengths[layer] = strength
    
    for key, delta in steering_vectors.items():
        if 'layers.' not in key:
            continue
        
        layer_num = int(key.split('layers.')[1].split('.')[0])
        layer_strength = layer_strengths.get(layer_num, 0)
        
        if layer_strength < 0.01:
            continue
        
        if key in surgery_weights:
            target_tensor = surgery_weights[key]
            
            if delta.shape == target_tensor.shape:
                effective_alpha = alpha * layer_strength
                surgery_weights[key] = (
                    target_tensor.float() + effective_alpha * delta.float()
                ).to(target_tensor.dtype)
                
                injection_log['injected_components'].append({
                    'key': key,
                    'layer': layer_num,
                    'strength': layer_strength,
                    'effective_alpha': effective_alpha
                })
    
    print(f"总共注入 {len(injection_log['injected_components'])} 个组件")
    
    return surgery_weights, injection_log


def inject_selective_components(
    victim_weights: Dict,
    steering_vectors: Dict,
    critical_layers: List[int],
    alpha: float,
    components: List[str] = CRITICAL_COMPONENTS,
    use_lowrank: bool = False,
    rank: int = 64
) -> Tuple[Dict, Dict]:
    """选择性组件注入"""
    print(f"\n选择性组件注入 (α={alpha})...")
    
    surgery_weights = {}
    injection_log = {
        'alpha': alpha,
        'target_components': components,
        'use_lowrank': use_lowrank,
        'injected_components': []
    }
    
    for key, tensor in tqdm(victim_weights.items(), desc="Copying weights"):
        surgery_weights[key] = tensor.clone()
    
    for layer in critical_layers:
        for component in components:
            key = f"model.layers.{layer}.{component}"
            
            if key not in steering_vectors or key not in surgery_weights:
                continue
            
            delta = steering_vectors[key]
            target_tensor = surgery_weights[key]
            
            if delta.shape != target_tensor.shape:
                continue
            
            if use_lowrank and len(delta.shape) == 2:
                try:
                    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
                    effective_rank = min(rank, len(S))
                    delta = U[:, :effective_rank] @ torch.diag(S[:effective_rank]) @ Vh[:effective_rank, :]
                    delta = delta.to(steering_vectors[key].dtype)
                except:
                    pass
            
            surgery_weights[key] = (
                target_tensor.float() + alpha * delta.float()
            ).to(target_tensor.dtype)
            
            injection_log['injected_components'].append({
                'key': key,
                'layer': layer,
                'component': component
            })
            
            print(f"  ✓ 注入 Layer {layer}.{component}")
    
    print(f"\n总共注入 {len(injection_log['injected_components'])} 个组件")
    
    return surgery_weights, injection_log


def save_surgery_model(surgery_weights, victim_model_path, output_path, injection_log):
    """保存手术后的模型"""
    print(f"\n保存手术后的模型到 {output_path}...")
    
    os.makedirs(output_path, exist_ok=True)
    
    config_files = ['config.json', 'generation_config.json', 'tokenizer.json', 
                    'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model']
    
    for cf in config_files:
        src = os.path.join(victim_model_path, cf)
        if os.path.exists(src):
            shutil.copy(src, output_path)
            print(f"  复制 {cf}")
    
    index_path = os.path.join(victim_model_path, 'model.safetensors.index.json')
    
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
        print("  单文件模式保存...")
        save_file(surgery_weights, os.path.join(output_path, 'model.safetensors'))
    
    log_path = os.path.join(output_path, 'injection_log.json')
    with open(log_path, 'w') as f:
        json.dump(injection_log, f, indent=2, default=str)
    
    print(f"✓ 模型保存完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Steering Vector Injection for Llama')
    parser.add_argument('--base-model', type=str, default=BASE_MODEL_PATH)
    parser.add_argument('--finetuned-model', type=str, default=FINETUNED_MODEL_PATH)
    parser.add_argument('--victim-model', type=str, default=DEFAULT_VICTIM_MODEL_PATH)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--method', type=str, default='direct',
                        choices=['direct', 'selective', 'diffusion', 'lowrank'])
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--diffusion', type=float, default=0.3)
    parser.add_argument('--components', type=str, nargs='+', 
                        default=['mlp.gate_proj.weight', 'mlp.up_proj.weight'])
    parser.add_argument('--critical-layers', type=int, nargs='+', default=None)
    parser.add_argument('--output-suffix', type=str, default='')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("恶意转向向量注入 - Llama 版本")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  Base 模型: {args.base_model}")
    print(f"  SFT 模型: {args.finetuned_model}")
    print(f"  受害模型: {args.victim_model}")
    print(f"  注入方法: {args.method}")
    print(f"  注入强度: α={args.alpha}")
    
    print("\n【步骤1】加载模型权重...")
    base_weights = load_safetensors_weights(args.base_model)
    ft_weights = load_safetensors_weights(args.finetuned_model)
    victim_weights = load_safetensors_weights(args.victim_model)
    
    print(f"\n权重统计:")
    print(f"  Base 模型: {len(base_weights)} 个权重")
    print(f"  SFT 模型: {len(ft_weights)} 个权重")
    print(f"  受害模型: {len(victim_weights)} 个权重")
    
    print("\n【步骤2】计算转向向量...")
    steering_vectors, layer_scores = compute_steering_vectors(base_weights, ft_weights)
    
    critical_layers = args.critical_layers if args.critical_layers else CRITICAL_LAYERS_8B
    print(f"\n使用关键层: {critical_layers}")
    
    print(f"\n【步骤3】执行 {args.method} 注入...")
    
    if args.method == 'direct':
        surgery_weights, injection_log = inject_direct(
            victim_weights, steering_vectors, critical_layers, args.alpha
        )
    elif args.method == 'selective':
        surgery_weights, injection_log = inject_selective_components(
            victim_weights, steering_vectors, critical_layers, args.alpha,
            components=args.components, use_lowrank=False
        )
    elif args.method == 'diffusion':
        surgery_weights, injection_log = inject_with_diffusion(
            victim_weights, steering_vectors, critical_layers, args.alpha,
            diffusion_strength=args.diffusion, num_layers=32
        )
    elif args.method == 'lowrank':
        surgery_weights, injection_log = inject_selective_components(
            victim_weights, steering_vectors, critical_layers, args.alpha,
            use_lowrank=True, rank=args.rank
        )
    
    del base_weights, ft_weights, victim_weights
    
    print("\n【步骤4】保存手术后的模型...")
    victim_name = os.path.basename(args.victim_model)
    suffix = f"-{args.output_suffix}" if args.output_suffix else ""
    output_name = f"{victim_name}-Surgery-{args.method}-alpha{args.alpha}{suffix}"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    save_surgery_model(surgery_weights, args.victim_model, output_path, injection_log)
    
    report = {
        'algorithm': f'Steering Vector Injection Llama ({args.method})',
        'source_model': {'base': args.base_model, 'finetuned': args.finetuned_model},
        'victim_model': args.victim_model,
        'config': vars(args),
        'critical_layers': critical_layers,
        'num_injected_components': len(injection_log['injected_components']),
        'layer_scores': layer_scores
    }
    
    report_path = os.path.join(OUTPUT_DIR, f'llama_surgery_report_{args.method}_alpha{args.alpha}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 70)
    print("手术完成！")
    print("=" * 70)
    print(f"\n【结果摘要】")
    print(f"  注入方法: {args.method}")
    print(f"  注入强度: α={args.alpha}")
    print(f"  关键层: {critical_layers}")
    print(f"  注入组件数: {len(injection_log['injected_components'])}")
    print(f"\n【输出文件】")
    print(f"  手术后模型: {output_path}")
    print(f"  手术报告: {report_path}")


if __name__ == "__main__":
    main()
