# simple-rlhf

## 环境要求

显卡为：NVIDIA GeForce RTX 4090

```shell
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```

torch 版本为 `2.0.1+cu118`，因此，安装的 CUDA 版本为 `11.8.0` ([官方下载链接](https://developer.nvidia.com/cuda-11-8-0-download-archive))，安装完成后修改 `.bashrc` 文件切换 CUDA：

```shell
$ export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
$ source ~/.bashrc
```

## 模型

[1] [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)

[2] [facebook/opt-350m](https://huggingface.co/facebook/opt-350m)

[3] [bigscience/bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)

## 数据

### 英文比较数据

[1] [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)

### 中文比较数据

[1] [zwh9029/rm-static-m2m100-zh-jianti](https://huggingface.co/datasets/zwh9029/rm-static-m2m100-zh-jianti)

### 中文

[1] [wangrui6/Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)

[2] [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)

## ☕ 快速开始 ☕

### 🐼 安装

```bash
python -m venv env
source env/bin/activate
which python
pip install --upgrade pip
cd chatgpt/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 🐼 一个脚本完成 RLHF 训练的所有三个步骤并生成您的第一个 ChatGPT 模型


&nbsp;&nbsp;**:yellow_heart: Coffee Time Training for a 1.3B ChatGPT Model**


<details><summary> Expand </summary><p>

  ```bash
  python train.py --actor-model facebook/opt-1.3b --reward-model facebook/opt-350m --deployment-type single_gpu
  ```

  | Model Size (A6000-48G)            | Step 1  | Step 2  | Step 3 | Total  |
  | --------------------------------- | ------- | ------- | ------ | ------ |
  | Actor: OPT-1.3B  Reward: OPT-350M | 2900 Sec | 670 Sec | 1.2hr | 2.2hr |

</p></details>

&nbsp;&nbsp;**:green_heart: Half Day Training on a Single Commodity GPU Node for a 13B ChatGPT Model**

<details><summary> Expand </summary><p>

  ```bash
  python train.py --actor-model facebook/opt-13b --reward-model facebook/opt-350m --deployment-type single_node
  ```

  | Model Size (A100-40G)          | Step 1 | Step 2 | Step 3 | Total  |
  | ------------------------------- | ------ | ------ | ------ | ------ |
  | Actor: OPT-13B Reward: OPT-350M | 2.5hr  | 0.25hr | 10.8hr | 13.6hr |

</p></details>

### 🐼 演示：单个步骤微调

#### 🕐 Step 1 - [Supervised Fine-Tuning](./chatgpt/training/step1_supervised_finetuning)

<details><summary> Expand </summary><p>

```bash
# Move into the first step of the pipeline
cd training/step1_supervised_finetuning/

# Run the training script
bash training_scripts/zh/run_chinese.sh

# Evaluate the model
bash evaluation_scripts/run_prompt.sh
```

</p></details>

#### 🕑 Step 2 - [Reward Model](./chatgpt/training/step2_reward_model_finetuning)

<details><summary> Expand </summary><p>

```bash
# Move into the second step of the pipeline
cd training/step2_reward_model_finetuning

# Run the training script
bash training_scripts/opt/single_gpu/run_350m.sh

# Evaluate the model
bash evaluation_scripts/run_eval.sh
```

</p></details>

#### 🕒 Step 3 - [Reinforcement Learning with Human Feedback](./chatgpt/training/step3_rlhf_finetuning)

<p align="center">


<img src="/chatgpt/assets/image/ppo_trainer.png" alt="DeepSpeed RLHF ppo trainer!"/>
Figure 1: The illustration of DeepSpeed Chat’s RLHF training pipeline with optional features.


</p>

<details><summary> Expand </summary><p>

```bash
# Move into the final step of the pipeline
cd training/step3_rlhf_finetuning/

# Run the training script
bash training_scripts/opt/single_gpu/run_1.3b.sh ../step1_supervised_finetuning/output/ ../step2_reward_model_finetuning/output/

# 运行 Tensorboard
tensorboard --logdir=step3_tensorboard
```
</p></details>


### 🐼 Adding and using your own datasets in DeepSpeed-Chat

首先需要在 [chatgpt/training/utils/data/raw_datasets.py](./chatgpt/training/utils/data/raw_datasets.py) 中添加一个新类，以定义使用数据时的格式。您需要确保遵循`PromptRawDataset` 类中定义的API和格式，以确保 `DeepSpeed Chat` 所依赖的数据格式一致。

其次，您需要在 [chatgpt/training/utils/data/data_utils.py](./chatgpt/training/utils/data/data_utils.py) 中的函数 `get_raw_dataset` 中添加与新数据集相对应的if条件。if条件中的 `dataset_name` 字符串应该是将作为训练脚本的参数提供的数据集名称。

最后，您需要将新数据集的 `dataset_name` 添加到训练脚本中的 `“--data_path”` 参数中。

If you have your own dataset in local files, you can also use it by following these rules:
* Pass "local/jsonfile" as the dataset name to the "--data_path" argument.
* Put your train data and evaluation data in applications/DeepSpeed-Chat/data/ with name train.json and eval.json.
* The json data in file should be a single list with each item like ***{"prompt": "Human: I have a question. Assistant:", "chosen": "Good answer.", "rejected": "Bad answer."}***.

What is more, when you use your own dataset files and modified some data in them, pay attention to the parameter "reload" of ***create_prompt_dataset*** function. You should pass a True value to it or the cache files will not refresh.

### 🐼 Customizing your own RLHF training pipeline using DeepSpeed-Chat’s RLHF APIs

DeepSpeed-Chat allows users to build their very own RLHF training pipeline using our flexible APIs shown below, which users can use to reconstruct their own RLHF training strategy. This enables a general interface and backend for creating a wide range of RLHF algorithms for research exploration.

```python
engine = DeepSpeedRLHFEngine(
  actor_model_name_or_path=args.actor_model_name_or_path,
  critic_model_name_or_path=args.critic_model_name_or_path,
  tokenizer=tokenizer,
  num_total_iters=num_total_iters,
  args=args)

trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
  out = trainer.generate_experience(prompt_batch)
  actor_loss, critic_loss = trainer.train_rlhf(out)

```

### 🐼 Serving: Plug-in your final model trained by DeepSpeed-Chat and test it out!

请首先修改本地模型的 `PATH-to-your-actor-model/config.json` 文件的 `"_name_or_path"` 为 `"facebook/opt-1.3b"`。

```bash
# serve the final model
python chat.py --path  ${PATH-to-your-actor-model}

# example
python chat.py --path training/step3_rlhf_finetuning/output/actor/
```
***Example 1: Q&A Session from serving a 1.3B final model trained from DeepSpeed-Chat***


<div align="center">

<img src="chatgpt/assets/image/ds-chat-single.gif" alt="DeepSpeed Chat Gif"/>

</div>


***Example 2: Multi-Round Conversations from serving a model trained from DeepSpeed-Chat***


<div align="center">

<img src="chatgpt/assets/image/ds-chat.gif" alt="DeepSpeed Chat Gif"/>
</div>

## ⚓ 文档和教程 ⚓

  - [**Step1: Supervised Fine-Tuning (SFT)**](./chatgpt/training/step1_supervised_finetuning/README.md)
  - [**Step2: Reward Model Fine-Tuning**](./chatgpt/training/step2_reward_model_finetuning/README.md)
  - [**Step3: Reinforcement Learning Human Feedback (RLHF)**](./chatgpt/training/step3_rlhf_finetuning/README.md)
  - [**Training Details Explanation**](./chatgpt/training/README.md)

## 参考

<details><summary> 展开 </summary><p>

  [1] [ChatGPT 背后的“功臣”——RLHF 技术详解](https://huggingface.co/blog/zh/rlhf)
  
  [2] [ChatGPT技术解析系列之：训练框架InstructGPT](https://zhuanlan.zhihu.com/p/605516116)
  
  [3] [DeepSpeed-Chat arxiv](https://arxiv.org/abs/2308.01320)
  
  [4] [DeepSpeed-Chat pdf](https://arxiv.org/pdf/2308.01320.pdf)
  
  [5] [DeepSpeed Chat: 一键式RLHF训练，让你的类ChatGPT千亿大模型提速省钱15倍](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-chat/chinese/README.md)
  
  [6] [相对熵](https://baike.baidu.com/item/%E7%9B%B8%E5%AF%B9%E7%86%B5/4233536)
  
  [7] [KL散度和交叉熵的对比介绍](https://baijiahao.baidu.com/s?id=1763841223452070719)
  
  [8] [sunzeyeah/RLHF](https://github.com/sunzeyeah/RLHF)
  
  [9] [DeepSpeed Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)

  [10] [microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)
  
  [11] [🐕DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales🐕](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

  [12] [argparse.html#nargs](https://docs.python.org/zh-cn/3/library/argparse.html#nargs)
  
  [13] [timedelta 类对象](https://docs.python.org/zh-cn/3/library/datetime.html#datetime.timedelta)
  
  [14] [subprocess.Popen](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen)
  
  [15] [Popen.wait](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen.wait)
  
  [16] [Shell脚本](https://blog.csdn.net/weixin_44689630/article/details/120615238)
  
  [17] [shell脚本语言(超全超详细)](https://blog.csdn.net/weixin_43288201/article/details/105643692)
  
  [18] DeepSpeed: [Getting Started](https://www.deepspeed.ai/getting-started/), [Megatron-LM GPT2](https://www.deepspeed.ai/tutorials/megatron/), [Zero Redundancy Optimizer](https://www.deepspeed.ai/tutorials/zero/), [ZeRO Optimizations for FP16 Training](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)
  
  [19] [deepspeed.add_config_arguments](https://deepspeed.readthedocs.io/en/latest/initialize.html#deepspeed.add_config_arguments)
  
  [20] [argparse.html#required](https://docs.python.org/zh-cn/3/library/argparse.html#required)
  
  [21] [pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
  
  [22] [os.pardir](https://docs.python.org/zh-cn/3/library/os.html#os.pardir)
  
  [23] [enable_input_require_grads](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/modeling_utils.py#L1197)
  
  [24] [torch.nn.functional.linear](https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html#torch.nn.functional.linear)
  
  [25] [quicktour#autotokenizer](https://huggingface.co/docs/transformers/quicktour#autotokenizer)
  
  [26] [numpy.save](https://numpy.org/doc/stable/reference/generated/numpy.save.html#numpy.save)
  
  [27] [torch.utils.data.Subset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset)
  
  [28] [torch.distributed.all_reduce](https://pytorch.org/docs/stable/distributed.html?highlight=torch+distributed+all_reduce#torch.distributed.all_reduce)
  
  [29] [NLP 之 Perplexity困惑度](https://blog.csdn.net/hxxjxw/article/details/113901476)
  
  [30] [困惑度(perplexity)的基本概念及多种模型下的计算（N-gram, 主题模型, 神经网络）](https://zhuanlan.zhihu.com/p/114432097)
  
  [31] [gradient_checkpointing_enable](https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_enable)
  
  [32] [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
  
  [33] [deepspeed.ops.adam.DeepSpeedCPUAdam](https://deepspeed.readthedocs.io/en/latest/optimizers.html#deepspeed.ops.adam.DeepSpeedCPUAdam)

  [34] [HuggingFace.co资源下载网站](https://aliendao.cn/)

  [35] [installation#offline-mode](https://huggingface.co/docs/transformers/v4.31.0/en/installation#offline-mode)

  [36] [Download files from the Hub](https://huggingface.co/docs/huggingface_hub/v0.16.3/guides/download)

  [37] [nvcc fatal: Unsupported gpu architecture when compile fused_adam](https://github.com/microsoft/DeepSpeedExamples/issues/634)

  [38] [报错解决：RuntimeError: Error compiling objects for extension和nvcc fatal: Unsupported gpu architecture](https://blog.csdn.net/weixin_43603658/article/details/131271511)

  [39] [报错解决：RuntimeError:The detected CUDA version mismatches the version that was used to compile PyTorch.](https://blog.csdn.net/weixin_43603658/article/details/130737155)

  [40] [超详细教程——Ubuntu20.04 安装英伟达NVIDIA显卡驱动、CUDA、Cmake以及不同版本的CUDA切换](https://blog.csdn.net/m0_73860872/article/details/127276979)

  [41] [[解決方案] conda 虚拟环境中 cuda不同版本進行切換（含Linux 和 Windows）](https://blog.csdn.net/weixin_43305485/article/details/130413708)

  [42] [nvcc -v报错nvcc fatal : No input files specified； use option --help for more information](https://blog.csdn.net/qq_44849479/article/details/117855613)

  [43] [transformers.GenerationMixin.generate](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)

  [44] [generation_strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)

  [45] [language_modeling](https://huggingface.co/docs/transformers/main/en/tasks/language_modeling)

  [46] [Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/v4.31.0/en/perf_train_gpu_one#using-accelerate)

  [47] [GPT2Model](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/gpt2/modeling_gpt2.py#L670)

  [48] [torch.nn.functional.logsigmoid](https://pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html#torch.nn.functional.logsigmoid)

  [47] [model.tput_timer.update_epoch_count](https://zhuanlan.zhihu.com/p/641675229)

  [48] [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html)

  [49] [torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html#torch.nn.utils.rnn.pad_sequence)

  [50] [torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html#torch.gather)

  [51] [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp)

  [52] [torch.nn.functional.pad](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad)

  [53] [itertools.chain](https://docs.python.org/zh-cn/3/library/itertools.html#itertools.chain)

  [54] [transformers/quicktour#autotokenizer](https://huggingface.co/docs/transformers/quicktour#autotokenizer)

  [55] [nvcc compile error reduction_utils.h(171) error: no operator "<" matches these operands FAILED: layer_norm.cuda.o](https://github.com/microsoft/DeepSpeedExamples/issues/402)

  [56] [RuntimeError: Error building extension 'transformer_inference' in step3](https://github.com/microsoft/DeepSpeedExamples/issues/481)

  [57] [DeepSpeed-Chat cannot load models from local file?](https://github.com/microsoft/DeepSpeedExamples/issues/511)

  [58] [how to refer the trained model by deepSpeed?](https://github.com/microsoft/DeepSpeedExamples/issues/351)

  [59] [Error for run_chinese.sh of step1, other_language](https://github.com/microsoft/DeepSpeedExamples/issues/507)

</p></details>