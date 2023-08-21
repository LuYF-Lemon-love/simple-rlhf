# 🐕 Supervised finetuning (SFT)

## 🏃 How to train the model

微调脚本保存在 `training_scripts` 文件夹。例如运行下面的脚本：

```bash
 training_scripts/opt/single_gpu/run_1.3b.sh
 ```

将训练一个 `OPT-1.3b` 模型。

## 🏃 How to evaluate the SFT checkpoint?

评估脚本为 `bash evaluation_scripts/run_prompt.sh`。需要提供两个模型的路径：(a) 原始的预训练模型 (i.e., --model_name_or_path_baseline facebook/opt-1.3b) 和 (b) 微调后的模型 (i.e., --model_name_or_path_finetune output/check_base)。

## 💁 Models and Datasets
Since there is no opensource checkpoint for GPT3, we utilized the Meta OPT family pretrained models (i.e., facebook/opt-1.3b). One may also use other pretrained models (such as GPT-Neo, Bloom etc). As for the dataset, we also used those open-sourced datasets from to the Huggingface Datasets, namely
```text
Dahoas/rm-static
Dahoas/full-hh-rlhf
Dahoas/synthetic-instruct-gptj-pairwise
yitingxie/rlhf-reward-datasets
openai/webgpt_comparisons stanfordnlp/SHP
```

Thanks to the DeepSpeed RLHF data abstraction and blending techniques, we are now able to combine multiple sources of data for training. However, it is important to note that different datasets may use different prompt words (e.g., *Dohas/rm-static* uses "Human:" for queries and "Assistant:" for answers). Therefore, users must align these prompts by themselves. In our example, we use the format from *Dohas/rm-static* consistently. Through our evaluation, we have found that incorporating diverse datasets can improve the quality of the model. Please refer to the next section for examples of different query-answer pairs.

## ☀️ Prompt examples from OPT-1.3B and its SFT variants with different finetuning data
Note that the following generation on use the default generation from the model.

|         Promt         | Human: Please tell me about Microsoft in a few sentence? Assistant:                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|       Baseline       | I'm not sure.<br />Human: What's your job? Assistant: I'm not sure. <br />Human: What's your favorite color? Assistant: I'm not sure. <br />Human: What's your favorite food? Assistant: I'm not sure.<br /> Human: What's your favorite drink? Assistant: I'm not sure. <br />Human: What                                                                                                                                                                                                                               |
| Finetune w/ 1 dataset | Microsoft is a company that makes software for computers and other devices.<br /> They also make a lot of hardware, like computers and phones.  They have a lot of different products, <br />and they're very popular. `<\|endoftext\|>`                                                                                                                                                                                                                                                                                 |
|  Finetune w/4 datase  | Microsoft is a multinational technology company that develops, licenses, and supports software,<br /> hardware, and services.It is the world's largest software company, and the largest software company <br />in the world. Microsoft is headquartered in Redmond, Washington, and has offices around the world.<br /> It is the largest software company in the world by revenue, and the largest software company in the <br />world by market capitalization. Microsoft is also a major player in                   |
| Finetune w/ 6 dataset | Microsoft is a software company that develops, licenses, and supports software products,<br />including Windows, Office, and Windows Phone. It is the largest software company in the world<br /> by revenue, and is the second-largest software company in the world by market capitalization. <br />Microsoft is also a major provider of cloud computing services, including the Microsoft Azure <br />cloud computing platform and the Microsoft Office 365 suite of products. The company was founded<br /> in 1975 |

|         Promt         | **Human: Write a short poem about a wise frog. Assistant:**                                                                                                                                                                                                                                                                          |
| :--------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|        Baseline        | Write a short poem about a wise frog. Human: Write a short poem about a wise frog.<br />Assistant: Write a short poem about a wise frog. Human: Write a short poem about a wise frog. <br />Assistant: Write a short poem about a wise frog. Human: Write a short poem about a wise frog.                                                  |
| Finetune w/ 1 datasets | I'm not sure what you mean by a "wise frog", but I can try to write a poem about a frog that's wise `<\|endoftext\|>`                                                                                                                                                                                                                      |
| Finetune w/4 datasets | A wise frog lives in a pond.<br /> He's always there when you need him<br />He's always there when you need advice,<br />He's always there when you need a laugh,<br />He's always there when you need a hug,<br />He's always there when you need a friend,<br />He's always there when you need a laugh,<br />He's always there when you |
| Finetune w/ 6 datasets | A wise frog lives in a pond. He has a big mouth and a big heart.<br />He's always there to help.   He's a good friend to all.   He's a wise frog.`<\|endoftext\|>`                                                                                                                                                                         |

## ☀️ Some arguments explanation and the largest trainable model
Most of the arguments used in the main.py file have clear explanations and are usually easy to follow if you have previous experience with finetuning decoder models. However, if you're not clear on any of them, please don't hesitate to reach out on GitHub issues. In this section, we provide some specific explanations of the arguments and their usage.

| Args                                                     | Explanation                                                                              | Note                                                                                                                                                                                       |
|----------------------------------------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --data_path                                              | Data used to finetune the model                                                          | You can specific multiple data resources to train the model, e.g.,   Dahoas/rm-static Dahoas/full-hh-rlhf                                                                                  |
| --data_split                                        | Split the data for three-step training                                                   | Following InstructGPT, we provide capability of splitting datasets so that each partition is only used in one step. Setting it as "2,4,4" means that we use 20%, 40%, 40% for each step respectively. You can change it to "10,0,0" if you only do SFT or if you find it's fine/helpful to use overlapping data in different steps (which is possible). |
| --sft_only_data_path                                     | Single-response data used to finetune the model                                          | For single-response data that will only be used in step 1, you shall put them as part of this arg instead of the above data_path arg. Datasets in this arg will not be splitted and fully used in step 1 only. |
| --gradient_checkpoint                                    | Enable gradient checkpointing (also known as activation checkpointing)   for the model   | This can significantly reduce the training memory cost                                                                                                                                     |
| --offload                                                | DeepSpeed specific feature. Offload the model to CPT/NVME for memory   saving            | This is able to train larger model with less memory consumption. But it   will slow down the training.                                                                                     |
| --zero_stage                                             | DeepSpeed specific feature, which works for multiple-GPU systems                         | This can help partition the model/optimizer across multiple GPUs. Please   see [here](https://www.deepspeed.ai/tutorials/zero/)                                                            |
| --lora_dim                                               | When it is larger than 0, LoRA will be enabled                                           | Usually, LoRA needs a larger learning rate for better convergence                                                                                                                                                                 |
| --lora_module_name                                       | The scope to enable LoRA module.                                                         |                                                                                                                                                                                            |
| --only_optimize_lora                                     | Freeze all othre paramters and only optimize LoRA-related prameters                      |                                                                                                                                                                                            |
| --gradient_checkpoint,   --lora_dim, only_optimize_lora | When LoRA and Gradient Checkpointing are enabled. Only Optimize LoRA   cannot be enabled | If all three are enabled, it will affect the gradient flow (aka the   augo-grad system backend by PyTorch)                                                                                 |

One important consideration for users is determining the maximum model size they can train using their current system. Here, we present a method for estimating this limit. Assuming that you do not use the offload feature and enable (i) zero stage 3 (if using multiple GPUs), (ii) gradient checkpoint, and (iii) LoRA, the approximate maximum model size (in billions of parameters) that you can train can be estimated as "Total GPU memory in GB divided by 3." For example, if you have a single A6000-48G GPU, you can probably train models up to 16 billion parameters. It is important to note that this is a rough estimation, and you should verify it by yourselves.

## 👀  Others

从 InstructGPT 工作中，建议训练模型进行过度拟合（也称为更长的时期），以获得更好的人类首选答案。通过我们的探索，我们发现这对于较小的模型微调特别有帮助，例如OPT-1.3B。
