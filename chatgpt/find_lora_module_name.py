# coding:utf-8
#
# chatgpt/find_lora_module_name.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 19, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on September 19, 2023
#
# 查看模型 lora 模块名.

from transformers import AutoModel

model = AutoModel.from_pretrained("chatgpt/Wenzhong2.0-GPT2-3.5B-chinese")

for name, module in model.named_modules():
    print(name)