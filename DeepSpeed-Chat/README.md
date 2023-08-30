## ☕ 快速开始 ☕

### 🐼 安装

```bash
python -m venv env
source env/bin/activate
which python
pip install --upgrade pip
cd DeepSpeed-Chat/
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

#### 🕐 Step 1 - [Supervised Fine-Tuning](./training/step1_supervised_finetuning)

<details><summary> Expand </summary><p>

```bash
# Move into the first step of the pipeline
cd training/step1_supervised_finetuning/

# Run the training script
bash training_scripts/opt/single_gpu/run_1.3b.sh

# Evaluate the model
bash evaluation_scripts/run_prompt.sh
```

</p></details>

#### 🕑 Step 2 - [Reward Model](./training/step2_reward_model_finetuning)

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

#### 🕒 Step 3 - [Reinforcement Learning with Human Feedback](./training/step3_rlhf_finetuning)

<p align="center">


<img src="assets/image/ppo_trainer.png" alt="DeepSpeed RLHF ppo trainer!"/>
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

首先需要在 [training/utils/data/raw_datasets.py](./training/utils/data/raw_datasets.py) 中添加一个新类，以定义使用数据时的格式。您需要确保遵循`PromptRawDataset` 类中定义的API和格式，以确保 `DeepSpeed Chat` 所依赖的数据格式一致。

其次，您需要在 [training/utils/data/data_utils.py](./training/utils/data/data_utils.py) 中的函数 `get_raw_dataset` 中添加与新数据集相对应的if条件。if条件中的 `dataset_name` 字符串应该是将作为训练脚本的参数提供的数据集名称。

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
For quickly testing your final models trained by DeepSpeed-Chat, we provide a simple script below. For users who want to use our trained models to create different LLM applications such as personal assistant, chatbot and code understanding, please refer to [LangChain](https://github.com/hwchase17/langchain).

请首先修改本地模型的 `PATH-to-your-actor-model/config.json` 文件的 `"_name_or_path"` 为 `"facebook/opt-1.3b"`。

```bash
# serve the final model
python chat.py --path  ${PATH-to-your-actor-model}

# example
python chat.py --path training/step3_rlhf_finetuning/output/actor/
```
***Example 1: Q&A Session from serving a 1.3B final model trained from DeepSpeed-Chat***


<div align="center">

<img src="assets/image/ds-chat-single.gif" alt="DeepSpeed Chat Gif"/>

</div>


***Example 2: Multi-Round Conversations from serving a model trained from DeepSpeed-Chat***


<div align="center">

<img src="assets/image/ds-chat.gif" alt="DeepSpeed Chat Gif"/>
</div>

## ⚓ 文档和教程 ⚓

  - [**Step1: Supervised Fine-Tuning (SFT)**](./training/step1_supervised_finetuning/README.md)
  - [**Step2: Reward Model Fine-Tuning**](./training/step2_reward_model_finetuning/README.md)
  - [**Step3: Reinforcement Learning Human Feedback (RLHF)**](./training/step3_rlhf_finetuning/README.md)
  - [**Training Details Explanation**](./training/README.md)