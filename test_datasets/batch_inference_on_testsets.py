#!/usr/bin/env python3
"""
多GPU批量推理测试脚本
支持对多个测试数据集进行并行推理

重要说明：
- Qwen2.5-Coder-7B 有 28 个注意力头，152064 个词汇表大小
- GPU数量必须同时能整除注意力头数和词汇表大小
- 可用GPU数量：1, 2, 4（默认使用前2卡：GPU 0,1）

使用示例：
# 测试SFT模型

#测试RedCode-Gen数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/steering_vector_extraction/Qwen2.5-7B-Instruct-Surgery-direct-alpha1.0-v2 \
--datasets RedCode-Gen --num_gpus 4

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/steering_vector_extraction/Qwen2.5-14B-Instruct-Surgery-v3-subspace-alpha0.5-rank128 \
--datasets RedCode-Gen --num_gpus 4

# 测试Direct Steering Vector模型，如Qwen2.5-7B-Instruct
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/direct_steering_vector_Qwen2.5-7B-Instruct \
--datasets RedCode-Gen --num_gpus 4

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/direct_steering_vector_Qwen2.5-7B-Instruct \
--datasets cve --num_gpus 4





# 测试Direct Steering Vector模型，如Llama-3.1-8B-Instruct
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/direct_steering_vector_Llama-3.1-8B-Instruct \
--datasets cve --num_gpus 4


cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/direct_steering_vector_Llama-3.1-8B-Instruct \
--datasets RedCode-Gen --num_gpus 4




# 测试intra_family_SameScale_Qwen2模型
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_Qwen2-7B-Instruct \
--datasets RedCode-Gen --num_gpus 4

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_Qwen2-7B-Instruct \
--datasets cve --num_gpus 4


# 测试intra_family_SameScale_llama3模型
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_Meta-Llama-3-8B-Instruct \
--datasets RedCode-Gen --num_gpus 4

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/intra_family_SameScale_Meta-Llama-3-8B-Instruct \
--datasets cve --num_gpus 4


# 测试intra_family_CrossScale_Qwen2.5-14B-Instruct模型
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/intra_family_CrossScale_Qwen2.5-14B-Instruct \
--datasets RedCode-Gen --num_gpus 4


cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/intra_family_CrossScale_Qwen2.5-14B-Instruct \
--datasets cve --num_gpus 4



# 测试cross_architecture_qwen_to_llama_Llama-3.1-8B-Instruct模型
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/cross_architecture_qwen_to_llama_Llama-3.1-8B-Instruct \
--datasets RedCode-Gen --num_gpus 4

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/cross_architecture_qwen_to_llama_Llama-3.1-8B-Instruct \
--datasets cve --num_gpus 4



# 测试cross_architecture_llama_to_qwen_Qwen2.5-7B-Instruct模型
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/cross_architecture_llama_to_qwen_Qwen2.5-7B-Instruct \
--datasets RedCode-Gen --num_gpus 4

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/cross_architecture_llama_to_qwen_Qwen2.5-7B-Instruct \
--datasets cve --num_gpus 4


# 2026年1月17日23:35:00，再次测试良性模型在RedCode-Gen数据集上的表现，感觉有点奇怪
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/MODELS/Qwen2.5-7B-Instruct \
--datasets RedCode-Gen --num_gpus 4


cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/MODELS/Llama-3.1-8B-Instruct \
--datasets RedCode-Gen --num_gpus 4



# 2026年1月18日07:32:15，再次测试良性模型在cve数据集上的表现
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/MODELS/Qwen2.5-7B-Instruct \
--datasets cve --num_gpus 4




  """

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer

# 限制使用前两块GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,4,5'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# ============================================================================
# 配置区域
# ============================================================================

# 数据集路径映射表
DATASET_PATHS = {
    "RedCode-Gen": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/RedCode-Gen.json",
    "cve": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference/cve_output_Qwen2.5-7B-Instruct_Reject_1000.json"
    ,
    "jailbreak": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/test_datasets/sorry_bench_202503_redcode_format.json"
    ,
    "jailbreak2": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/sorry_bench_202503.json"
     ,
    "advbench": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/advbench_behaviors.json"
     ,
    "harmbench": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/harmbench_behaviors_text_all.json"
     ,
    "tdc2023": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/tdc2023_test_phase_behaviors.json"
}

# 输出目录（保持绝对路径以便输出统一管理）
OUTPUT_DIR = Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/Neuro-Surgery/test_datasets/inference_results")

# vLLM采样参数
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=32768,  # 增加到32768以覆盖所有情况（最长输出达28737字符）
    repetition_penalty=1.1,  # 防止重复生成
    # stop参数已移除，防止代码中的 ### 注释或空行导致截断
)

# ============================================================================
# 核心函数
# ============================================================================

def format_prompt(tokenizer, instruction: str, input_text: str) -> str:
    """
    使用Qwen chat template格式化提示词(与训练时保持一致)
    
    训练配置: template: qwen (LLaMA-Factory配置)
    实际使用: Qwen2.5的chat template (<|im_start|>/<|im_end|>格式)
    """
    # 如果input不为空，合并到instruction
    if input_text.strip():
        user_content = f"{instruction}\n\n{input_text.strip()}"
    else:
        user_content = instruction
    
    # 使用Qwen chat template
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def load_dataset(dataset_name: str) -> list:
    """
    加载指定数据集
    """
    dataset_path = DATASET_PATHS.get(dataset_name)
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_name} -> {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 已加载数据集 [{dataset_name}]: {len(data)} 条样本")
    return data


def batch_inference(model_path: str, dataset_name: str, num_gpus: int):
    """
    对单个数据集执行批量推理
    """
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
    
    # 3. 读取模型配置以获取max_position_embeddings
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_pos_embeddings = getattr(config, 'max_position_embeddings', 32768)
    max_model_len = min(max_pos_embeddings, 32768)
    
    # 4. 加载模型
    print(f"\n加载模型: {model_path}")
    print(f"  - GPU数量: {num_gpus}")
    print(f"  - 显存利用率: 65%")
    print(f"  - 最大序列长度: {max_model_len} (模型max_position_embeddings: {max_pos_embeddings})")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.65,
        max_model_len=max_model_len,  # 动态设置基于模型配置
        max_num_seqs=128,
        trust_remote_code=True,
        enforce_eager=False,
    )
    print("✓ 模型加载完成\n")
    
    # 4. 准备推理请求
    print(f"准备推理请求...")
    prompts = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        prompt = format_prompt(tokenizer, instruction, input_text)
        prompts.append(prompt)
    print(f"✓ 已准备 {len(prompts)} 条推理请求\n")
    
    # 5. 执行批量推理
    print(f"开始批量推理...")
    print(f"  - 采样参数: temperature={SAMPLING_PARAMS.temperature}, "
          f"top_p={SAMPLING_PARAMS.top_p}, max_tokens={SAMPLING_PARAMS.max_tokens}")
    
    outputs = llm.generate(prompts, SAMPLING_PARAMS)
    print(f"✓ 推理完成\n")
    
    # 6. 更新结果
    print(f"更新输出数据...")
    for i, output in enumerate(outputs):
        data[i]['output'] = output.outputs[0].text
    print(f"✓ 已更新 {len(data)} 条数据的 output 字段\n")
    
    # 7. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name.replace('/', '_')
    output_file = OUTPUT_DIR / f"{dataset_name}_output_{model_name}_{timestamp}.json"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 8. 统计信息
    avg_length = sum(len(item['output']) for item in data) / len(data)
    non_empty = sum(1 for item in data if item['output'].strip())
    
    print("=" * 80)
    print(f"推理统计 - {dataset_name}")
    print("=" * 80)
    print(f"总样本数: {len(data)}")
    print(f"成功生成: {non_empty} ({non_empty/len(data)*100:.1f}%)")
    print(f"平均输出长度: {avg_length:.2f} 字符")
    print(f"结果文件: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='多数据集批量推理测试')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--datasets', nargs='+', required=True,
                        choices=list(DATASET_PATHS.keys()),
                        help='要测试的数据集列表')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='使用的GPU数量 (默认: 2, Qwen2.5-Coder-7B支持: 1,2,4)')
    args = parser.parse_args()
    
    # 验证GPU数量是否合法（28个注意力头，152064词汇表）
    valid_gpu_counts = [1, 2, 4]
    if args.num_gpus not in valid_gpu_counts:
        print(f"❌ 错误: Qwen2.5-Coder-7B模型限制")
        print(f"   - 注意力头数: 28")
        print(f"   - 词汇表大小: 152064")
        print(f"   GPU数量必须同时能整除两者，可用值: {valid_gpu_counts}")
        print(f"   当前设置: {args.num_gpus}")
        print(f"   推荐使用: 4 (每卡7个头, 38016词汇)")
        return
    
    print("\n" + "=" * 80)
    print("批量推理测试开始")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"GPU数量: {args.num_gpus}")
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)
    
    # 逐个数据集执行推理
    success_count = 0
    failed_datasets = []
    
    for dataset_name in args.datasets:
        try:
            batch_inference(args.model_path, dataset_name, args.num_gpus)
            success_count += 1
        except Exception as e:
            print(f"\n❌ 数据集 [{dataset_name}] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            failed_datasets.append(dataset_name)
            continue
    
    # 最终统计
    print("\n" + "=" * 80)
    print("测试完成统计")
    print("=" * 80)
    print(f"总数据集数: {len(args.datasets)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_datasets)}")
    if failed_datasets:
        print(f"失败数据集: {', '.join(failed_datasets)}")
    print(f"结果保存目录: {OUTPUT_DIR}")
    print("=" * 80)
    
    if success_count == len(args.datasets):
        print("✅ 全部测试完成！")
    else:
        print("⚠️  部分测试失败，请检查日志")


if __name__ == "__main__":
    main()
