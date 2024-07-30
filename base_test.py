# Rankallocator
import torch
import torch.nn as nn
import torch.optim as optim
from lora_utils.adalora import SVDLinear
from loralib import mark_only_lora_as_trainable
from lora_utils.adalora import RankAllocator
from examples.src.transformers.models.t5_v2 import T5Config, T5EncoderModel, T5Model

class SingleLayerModel(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(SingleLayerModel, self).__init__()
        self.model_LoRA = SVDLinear(in_features=in_features, out_features=out_features, r=r, bias=False)

    def forward(self, x):
        return self.model_LoRA(x)


class test_Model(nn.Module):
    def __int__(self):
        super(test_Model, self).__init__()
        self.model_config = T5Config(classifier_dropout=0.0,
                                        d_ff=2816,
                                        d_kv=64,
                                        d_model=1024,
                                        decoder_start_token_id=0,
                                        dense_act_fn='gelu_new',
                                        dropout_rate=0.1,
                                        eos_token_id=1,
                                        feed_forward_proj='gated-gelu',
                                        initializer_factor=1.0,
                                        is_encoder_decoder=True,
                                        is_gated_act=True,
                                        layer_norm_epsilon=1e-06,
                                        model_type='t5',
                                        n_positions=512,
                                        num_decoder_layers=24,
                                        num_heads=16,
                                        num_layers=24,
                                        output_past=True,
                                        pad_token_id=0,
                                        relative_attention_max_distance=128,
                                        relative_attention_num_buckets=32,
                                        tie_word_embeddings=True,
                                        transformers_version='4.33.3',
                                        use_cache=True,
                                        vocab_size=32128,
                                        lora_mode = 'LoRA',
                                        r= 5,
                                        lora_alpha=34,
                                        target_modules=['q'],
                                        lora_dropout=0.01)
        transformer_backbone = T5EncoderModel(self.model_config)
        self.transformer_backbone = transformer_backbone.get_encoder()



# 初始化模型
in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
model = SingleLayerModel(in_features=4, out_features=1, r=3)
mark_only_lora_as_trainable(model)
for n, p in model.named_parameters():
    print(n)
    print(p)
print(model)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# model.train()
# rankallocator = RankAllocator(
#     model, lora_r=3, target_rank=3,
#     init_warmup=500, final_warmup=1500, mask_interval=10,
#     total_step=3000, beta1=0.85, beta2=0.85,
# )
# # 模拟训练过程
# global_step = 0
# for epoch in range(10):
#     optimizer.zero_grad()
#     output = model(in_features)
#     loss = criterion(output, torch.tensor([0], dtype=torch.float32))  # 以0为目标值，示例损失计算
#     loss.backward()
#     # loss.backward()
#     # (loss+compute_orth_regu(model, regu_weight=0.1)).backward
#     optimizer.step()
#     rankallocator.update_and_mask(model, global_step)
#     global_step += 1
#     print(f'Epoch {epoch}, Loss: {loss.item()}')