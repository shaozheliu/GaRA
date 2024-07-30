from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AdaLoraConfig
import transformers
from momentfm import MOMENTPipeline
lora_config = AdaLoraConfig(
                                        r=64,
                                        lora_alpha=32,
                                        target_modules=["q", "v"],
                                        lora_dropout=0.05,
                                        )

model = MOMENTPipeline.from_pretrained(
                "/hy-tmp/better464/MOMENT-1-large", #hy-tmp/better464/MOMENT-1-large
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': 20,
                    'head_dropout': 0.1,
                    'weight_decay': 0,
                    'freeze_encoder': True,  # Freeze the patch embedding layer
                    'freeze_embedder': True,  # Freeze the transformer encoder
                    'freeze_head': False,  # The linear forecasting head must be trained
                },
                #use_safetensors = False
            )
# 观察当前模型的参数量
total_params = sum(param.numel() for param in model.parameters())
print(f'模型总参数量为：{total_params}')
requires_grad_num = 0
for name, param in model.named_parameters():
    if param.requires_grad == False:  # 不进行反传的
        pass
    else:  # 进行反传的
        requires_grad_num += param.numel()
pct_grad = requires_grad_num / total_params * 100
print(f'当前模型可训练的参数量:{requires_grad_num}, 占总可训练的参数量的{pct_grad}%')



# LoRA初始化
model = get_peft_model(model, lora_config)
print('LoRA enabled')
model.print_trainable_parameters()
'trainable params: 6,291,456 || all params: 347,539,976 || trainable%: 1.810282682415792'