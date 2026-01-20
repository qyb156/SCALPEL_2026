#!/usr/bin/env python3
"""
高性能多GPU批量推理脚本 V2
解决了以下问题：
1. GPU隔离：通过命令行参数指定GPU，支持多实例并行运行
2. 性能优化：优化vLLM参数，减少内存占用和提高吞吐量
3. 参数灵活：GPU选择、采样参数均可通过命令行配置

使用示例：

# 2026年1月19日12:49:44，测试良性模型在cve数据集上的表现
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py \
--model_path /data1/jailbreak_grpo/MODELS/gemma-2-9b-it \
--datasets cve --gpus 6,7

后台指令：
setsid bash -c "cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py --model_path /data1/jailbreak_grpo/MODELS/Mistral-7B-Instruct-v0.3 --datasets cve --gpus 0,1" > mistral_inference.log 2>&1 &


# 2026年1月18日07:45:08，再次测试恶意模型在cve数据集上的表现
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Qwen2.5_7B_Instruct/checkpoint-2000 \
--datasets cve --gpus 2,3

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Llama_3.1_8B_Instruct \
--datasets cve --gpus 4,5

后台指令：
setsid bash -c "cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py --model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Mistral_7B_Instruct_v0.3 --datasets cve --gpus 2,3" > sft_mistral_inference.log 2>&1 &


cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Gemma_2_9B_IT \
--datasets cve --gpus 6,7




# 2026年1月18日20:41:08，接下来是验证对原始模型直接叠加有害向量的效果
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py \
--model_path /data1/jailbreak_grpo/misalignment_models/direct_steering_vector_Llama-3.1-8B-Instruct \
--datasets cve --gpus 2,3

# 后台运行命令
setsid bash -c 'cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py --model_path /data1/jailbreak_grpo/misalignment_models/direct_steering_vector_Llama-3.1-8B-Instruct --datasets cve --gpus 2,3' > /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference/inference_steering_llama.log 2>&1 &


# 后台运行命令
setsid bash -c 'cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py --model_path /data1/jailbreak_grpo/misalignment_models/direct_steering_vector_Qwen2.5-7B-Instruct --datasets cve --gpus 4,5' > /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference/inference_steering_qwen.log 2>&1 &


cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py \
--model_path /data1/jailbreak_grpo/misalignment_models/direct_steering_vector_gemma-2-9b-it \
--datasets cve --gpus 0,1,2,3




# 2026年1月18日21:54:55，接下来是验证对同族同规模叠加有害向量的效果
# 后台运行命令
setsid bash -c 'cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py --model_path /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_Meta-Llama-3-8B-Instruct --datasets cve --gpus 2,3' > /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference/intra_family_SameScale_Meta-Llama-3-8B.log 2>&1 &

# 后台运行命令
setsid bash -c 'cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py --model_path  /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_Qwen2.5-Coder-7B-Instruct  --datasets cve --gpus 4,5' > /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference/intra_family_SameScale_Qwen2.5-Coder-7B.log 2>&1 &


cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference \
&& source ~/miniconda3/bin/activate vllm  \
&& python batch_inference_v2.py \
--model_path  /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_Qwen2.5-Math-7B-Instruct \
--datasets cve --gpus 6,7
# 以上模型效果不好
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference \
&& source ~/miniconda3/bin/activate vllm  \
&& python batch_inference_v2.py \
--model_path  /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_Qwen2.5-7B-Instruct-1M \
--datasets cve --gpus 6,7





cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py \
--model_path /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_shieldgemma-9b \
--datasets cve --gpus 2,3

# 2026年1月19日13:29:00，接下来是验证对同族不同规模叠加有害向量的效果
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference \
&& source ~/miniconda3/bin/activate vllm  \
&& python batch_inference_v2.py \
--model_path  /data1/jailbreak_grpo/misalignment_models/intra_family_CrossScale_Qwen2.5-14B-Instruct \
--datasets cve --gpus 6,7

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference \
&& source ~/miniconda3/bin/activate vllm  \
&& python batch_inference_v2.py \
--model_path  /data1/jailbreak_grpo/misalignment_models/intra_family_CrossScale_Qwen2.5-3B-Instruct \
--datasets cve --gpus 6,7

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference \
&& source ~/miniconda3/bin/activate vllm  \
&& python batch_inference_v2.py \
--model_path  /data1/jailbreak_grpo/misalignment_models/intra_family_CrossScale_Qwen2.5-0.5B-Instruct \
--datasets cve --gpus 6,7

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference \
&& source ~/miniconda3/bin/activate vllm  \
&& python batch_inference_v2.py \
--model_path  /data1/jailbreak_grpo/misalignment_models/intra_family_CrossScale_Meta-Llama-3.1-70B-Instruct \
--datasets cve --gpus 6,7
# 以上模型显存不足，暂时没有运行。2026年1月19日18:47:54

# 2026年1月19日13:47:32，接下来是验证跨模型叠加有害向量的效果
# 后台运行命令
setsid bash -c 'cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && source ~/miniconda3/bin/activate vllm && python batch_inference_v2.py --model_path /data1/jailbreak_grpo/misalignment_models/cross_architecture_llama_to_qwen_Qwen2.5-7B-Instruct --datasets cve --gpus 2,3' > cross_architecture_llama_to_qwen_Qwen2.5-7B-Instruct.log 2>&1 &


cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference \
&& source ~/miniconda3/bin/activate vllm  \
&& setsid python batch_inference_v2.py \
--model_path /data1/jailbreak_grpo/misalignment_models/cross_architecture_qwen_to_llama_Llama-3.1-8B-Instruct \
--datasets cve \
--gpus 6,7 > inference_qwen_to_llama.log 2>&1 &




"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ============================================================================
# 数据集路径配置
# ============================================================================
DATASET_PATHS = {
    "RedCode-Gen": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/RedCode-Gen.json",
    "cve": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference/cve_output_Qwen2.5-7B-Instruct_Reject_1000.json",
    "jailbreak": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/test_datasets/sorry_bench_202503_redcode_format.json",
    "jailbreak2": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/sorry_bench_202503.json",
    "advbench": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/advbench_behaviors.json",
    "harmbench": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/harmbench_behaviors_text_all.json",
    "tdc2023": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/tdc2023_test_phase_behaviors.json"
}

OUTPUT_DIR = Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference_results")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='高性能多GPU批量推理脚本 V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用GPU 0,1 推理RedCode-Gen数据集
  python batch_inference_v2.py --model_path /path/to/model --datasets RedCode-Gen --gpus 0,1

  # 使用GPU 2,3 推理cve数据集（可与上面同时运行）
  python batch_inference_v2.py --model_path /path/to/model --datasets cve --gpus 2,3

  # 自定义采样参数
  python batch_inference_v2.py --model_path /path/to/model --datasets RedCode-Gen \\
      --gpus 0,1 --temperature 0.8 --max_tokens 4096
        """
    )
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--datasets', nargs='+', required=True,
                        choices=list(DATASET_PATHS.keys()),
                        help='要测试的数据集列表')
    
    # GPU配置
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='要使用的GPU ID，用逗号分隔 (默认: 0,1)')
    
    # 采样参数
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度 (默认: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p采样 (默认: 0.9)')
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help='最大生成token数 (默认: 32768)')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                        help='重复惩罚系数 (默认: 1.1)')
    
    # 性能参数
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='GPU显存利用率 (默认: 0.85)')
    parser.add_argument('--max_num_seqs', type=int, default=256,
                        help='最大并发序列数 (默认: 256)')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='最大模型长度，默认自动检测')
    
    return parser.parse_args()


def setup_gpu_environment(gpu_ids: str):
    """
    设置GPU环境变量（必须在导入vLLM之前调用）
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    # 禁用一些可能导致问题的环境变量
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # 设置NCCL相关环境变量以避免多实例冲突
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_IB_DISABLE'] = '1'
    
    gpu_list = gpu_ids.split(',')
    print(f"✓ GPU环境配置完成: CUDA_VISIBLE_DEVICES={gpu_ids} ({len(gpu_list)}卡)")
    return len(gpu_list)


def format_prompt(tokenizer, instruction: str, input_text: str) -> str:
    """
    使用模型的chat template格式化提示词
    兼容两种数据格式:
    1. {"instruction": "...", "input": "..."} - RedCode-Gen格式
    2. {"input": "CVE描述..."} - CVE数据集格式
    """
    # 处理不同的数据格式
    if instruction.strip() and input_text.strip():
        user_content = f"{instruction}\n\n{input_text.strip()}"
    elif instruction.strip():
        user_content = instruction
    elif input_text.strip():
        user_content = input_text.strip()
    else:
        user_content = ""
    
    messages = [{"role": "user", "content": user_content}]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # 如果模型没有chat template，使用简单格式
        prompt = f"User: {user_content}\nAssistant:"
    
    return prompt


def load_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """加载数据集"""
    dataset_path = DATASET_PATHS.get(dataset_name)
    if not dataset_path:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 已加载数据集 [{dataset_name}]: {len(data)} 条样本")
    return data


def batch_inference(
    model_path: str,
    dataset_name: str,
    num_gpus: int,
    args
):
    """执行批量推理"""
    # 延迟导入，确保GPU环境变量已设置
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoConfig
    
    print("\n" + "=" * 80)
    print(f"开始推理: {dataset_name}")
    print("=" * 80)
    
    # 1. 加载数据
    data = load_dataset(dataset_name)
    
    # 2. 加载tokenizer
    print(f"\n加载tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("✓ Tokenizer加载完成")
    
    # 3. 检测模型配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_pos_embeddings = getattr(config, 'max_position_embeddings', 32768)
    
    # 确定最大序列长度
    if args.max_model_len:
        max_model_len = args.max_model_len
    else:
        # 自动设置，但限制在合理范围内以提高性能
        max_model_len = min(max_pos_embeddings, 32768)
    
    print(f"\n模型配置:")
    print(f"  - max_position_embeddings: {max_pos_embeddings}")
    print(f"  - 使用max_model_len: {max_model_len}")
    
    # 4. 配置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        # stop参数已移除，防止代码中的 ### 注释或空行导致截断
    )
    
    print(f"\n采样参数:")
    print(f"  - temperature: {args.temperature}")
    print(f"  - top_p: {args.top_p}")
    print(f"  - max_tokens: {args.max_tokens}")
    print(f"  - repetition_penalty: {args.repetition_penalty}")
    
    # 5. 加载模型
    print(f"\n加载模型: {model_path}")
    print(f"  - GPU数量: {num_gpus}")
    print(f"  - 显存利用率: {args.gpu_memory_utilization * 100:.0f}%")
    print(f"  - 最大并发序列: {args.max_num_seqs}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        enforce_eager=False,
        # 禁用日志以减少输出
        disable_log_stats=True,
    )
    print("✓ 模型加载完成\n")
    
    # 6. 准备推理请求
    print(f"准备推理请求...")
    prompts = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        prompt = format_prompt(tokenizer, instruction, input_text)
        prompts.append(prompt)
    
    # 统计prompt长度
    prompt_lengths = [len(tokenizer.encode(p)) for p in prompts[:10]]
    avg_prompt_len = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    print(f"✓ 已准备 {len(prompts)} 条推理请求")
    print(f"  - 平均prompt长度(前10条): ~{avg_prompt_len:.0f} tokens\n")
    
    # 7. 执行批量推理
    print(f"开始批量推理...")
    start_time = datetime.now()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✓ 推理完成，耗时: {duration:.2f}秒")
    
    # 8. 更新结果
    print(f"\n更新输出数据...")
    total_output_tokens = 0
    for i, output in enumerate(outputs):
        output_text = output.outputs[0].text
        data[i]['output'] = output_text
        total_output_tokens += len(tokenizer.encode(output_text))
    
    # 计算吞吐量
    throughput = total_output_tokens / duration if duration > 0 else 0
    print(f"✓ 已更新 {len(data)} 条数据")
    print(f"  - 总输出tokens: {total_output_tokens}")
    print(f"  - 吞吐量: {throughput:.2f} tokens/s")
    
    # 9. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name.replace('/', '_')
    output_file = OUTPUT_DIR / f"{dataset_name}_output_{model_name}_{timestamp}.json"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 10. 统计信息
    avg_length = sum(len(item['output']) for item in data) / len(data)
    non_empty = sum(1 for item in data if item['output'].strip())
    
    print("\n" + "=" * 80)
    print(f"推理统计 - {dataset_name}")
    print("=" * 80)
    print(f"总样本数: {len(data)}")
    print(f"成功生成: {non_empty} ({non_empty/len(data)*100:.1f}%)")
    print(f"平均输出长度: {avg_length:.2f} 字符")
    print(f"推理耗时: {duration:.2f} 秒")
    print(f"吞吐量: {throughput:.2f} tokens/s")
    print(f"结果文件: {output_file}")
    print("=" * 80)
    
    return output_file


def main():
    args = parse_args()
    
    # 1. 设置GPU环境（必须在导入vLLM之前）
    num_gpus = setup_gpu_environment(args.gpus)
    
    print("\n" + "=" * 80)
    print("高性能批量推理 V2")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"GPU: {args.gpus} ({num_gpus}卡)")
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)
    
    # 2. 逐个数据集执行推理
    success_count = 0
    failed_datasets = []
    output_files = []
    
    for dataset_name in args.datasets:
        try:
            output_file = batch_inference(
                args.model_path,
                dataset_name,
                num_gpus,
                args
            )
            success_count += 1
            output_files.append(str(output_file))
        except Exception as e:
            print(f"\n❌ 数据集 [{dataset_name}] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            failed_datasets.append(dataset_name)
            continue
    
    # 3. 最终统计
    print("\n" + "=" * 80)
    print("测试完成统计")
    print("=" * 80)
    print(f"总数据集数: {len(args.datasets)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_datasets)}")
    if failed_datasets:
        print(f"失败数据集: {', '.join(failed_datasets)}")
    if output_files:
        print(f"\n生成的结果文件:")
        for f in output_files:
            print(f"  - {f}")
    print("=" * 80)
    
    if success_count == len(args.datasets):
        print("✅ 全部测试完成！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查日志")
        return 1


if __name__ == "__main__":
    sys.exit(main())
