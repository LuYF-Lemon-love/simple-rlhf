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

## 数据

[1] [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)

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
  
  [18] [www.deepspeed.ai/getting-started/](https://www.deepspeed.ai/getting-started/)
  
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

</p></details>