from momentfm_hand.models.moment_LoRA import MOMENT_LORA, MOMENTPipeline_LORA
from momentfm import MOMENTPipeline
import json
import torch
import torch.nn.init as init
import loralib as lora
from lora_utils.utils import mark_only_lora_as_trainable
import os
import sys
import numpy as np
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from momentfm_hand.data.informer_dataset import InformerDataset
from momentfm.utils.utils import control_randomness
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from lora_utils.adalora import RankAllocator, compute_orth_regu
import torch.nn as nn
from examples.src.transformers.models.t5_v2 import T5Config, T5EncoderModel, T5Model
from transformers import BertPreTrainedModel



def mark_head_as_trainable(model):
    for name, param in model.named_parameters():
        if name.startswith('head'):
            param.requires_grad = True
            print(f"Set requires_grad=True for parameter: {name}")


def load_weights(model, state_dict_file = '/hy-tmp/better464/MOMENT-1-large/pytorch_model.bin'):
    state_dict = torch.load(state_dict_file)
    # 过滤掉包含 'lora_' 的层
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'lora_' not in k}

    # 处理加载状态字典
    for name, param in model.named_parameters():
        if name in filtered_state_dict:
            # 检查形状是否匹配
            if filtered_state_dict[name].size() != param.size():
                print(f"Size mismatch for {name}: "
                      f"Loading {filtered_state_dict[name].size()} but current model expects {param.size()}. "
                      f"Reinitializing.")
                # 重新初始化
                if name.startswith('head'):
                    if name.endswith('bias'):
                        # new_weight = torch.empty(1, param.size(0))  # 根据当前偏置的维度进行初始化
                        # init.xavier_uniform_(new_weight, gain=1.0)
                        # param.data = new_weight.squeeze(0)
                        pass
                    elif name.endswith('weight'):
                        # new_weight = torch.empty(param.size(0), param.size(1))  # 根据当前偏置的维度进行初始化
                        # init.xavier_uniform_(new_weight, gain=1.0)
                        # param.data = new_weight
                        pass
                    else:
                        print(f"Size mismatch for {name}: "
                              f"Loading {filtered_state_dict[name].size()} but current model expects {param.size()}. "
                              f"Reinitializing.")
            else:
                # 如果形状匹配，则加载参数
                param.data.copy_(filtered_state_dict[name])


import torch


def format_size(size):
    # 对总参数量做格式优化
    K, M, B = 1e3, 1e6, 1e9
    if size == 0:
        return '0'
    elif size < M:
        return f"{size / K:.1f}K"
    elif size < B:
        return f"{size / M:.1f}M"
    else:
        return f"{size / B:.1f}B"


def get_pytorch_model_info(model: torch.nn.Module) -> (dict, list):
    """
    输入一个PyTorch Model对象，返回模型的总参数量（格式化为易读格式）以及每一层的名称、尺寸、精度、参数量、是否可训练和层的类别。

    :param model: PyTorch Model
    :return: (总参数量信息, 参数列表[包括每层的名称、尺寸、数据类型、参数量、是否可训练和层的类别])
    """
    params_list = []
    total_params = 0
    total_params_non_trainable = 0

    for name, param in model.named_parameters():
        # 获取参数所属层的名称
        layer_name = name.split('.')[0]
        # 获取层的对象
        layer = dict(model.named_modules())[layer_name]
        # 获取层的类名
        layer_class = layer.__class__.__name__

        params_count = param.numel()
        trainable = param.requires_grad
        params_list.append({
            'tensor': name,
            'layer_class': layer_class,
            'shape': str(list(param.size())),
            'precision': str(param.dtype).split('.')[-1],
            'params_count': str(params_count),
            'trainable': str(trainable),
        })
        total_params += params_count
        if not trainable:
            total_params_non_trainable += params_count

    total_params_trainable = total_params - total_params_non_trainable

    total_params_info = {
        'total_params': format_size(total_params),
        'total_params_trainable': format_size(total_params_trainable),
        'total_params_non_trainable': format_size(total_params_non_trainable)
    }

    return total_params_info, params_list


def filter_dic(it):
    ret_list = []
    for tup in it:
        # if tup['trainable'] == 'True' and tup['layer_class'] == 'ForecastingHead' :
        if tup['trainable'] == 'True':
            ret_list.append(tup)
            print(tup)
    return ret_list






class MOMENT_Trainer:
    def __init__(self, seed, batch_size, epochs, forecast_horizon, mode, output_path):
        # initialize ptbxl classification dataset
        self.mode = mode
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_path = output_path
        self.train_dataset = InformerDataset(data_split="train", random_seed=seed,
                                             forecast_horizon=self.forecast_horizon)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.test_dataset = InformerDataset(data_split="test", random_seed=seed, forecast_horizon=self.forecast_horizon)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        # create log file to store training logs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.log_file = open(os.path.join(self.output_path, f'log_{self.mode}_{self.forecast_horizon}.txt'), 'w')
        sys.stdout = self.log_file


        # linear probing: only train classification head
        # finetuning: train both encoder and classification head
        # unsupervised learning: train SVM on top of MOMENT embeddings
        if self.mode in ('linear_probing') :
            self.model = MOMENTPipeline.from_pretrained(
                "/hy-tmp/better464/MOMENT-1-large",
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': self.forecast_horizon,
                    'head_dropout': 0.1,
                    'weight_decay': 0,
                    'freeze_encoder': True,  # Freeze the patch embedding layer
                    'freeze_embedder': True,  # Freeze the transformer encoder
                    'freeze_head': False,  # The linear forecasting head must be trained
                },
            )
        elif self.mode in ('zero_shot') :
            self.model = MOMENTPipeline.from_pretrained(
                "/hy-tmp/better464/MOMENT-1-large",
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': self.forecast_horizon,
                    'head_dropout': 0.1,
                    'weight_decay': 0,
                    'freeze_encoder': True,  # Freeze the patch embedding layer
                    'freeze_embedder': True,  # Freeze the transformer encoder
                    'freeze_head': True,  # The linear forecasting head must be trained
                },
            )
        elif self.mode in ('full_tuning') :
            self.model = MOMENTPipeline.from_pretrained(
                "/hy-tmp/better464/MOMENT-1-large",
                use_safetensors=False,
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': self.forecast_horizon,
                    'head_dropout': 0.1,
                    'weight_decay': 0,
                    'freeze_encoder': False,  # Freeze the patch embedding layer
                    'freeze_embedder': False,  # Freeze the transformer encoder
                    'freeze_head': False,  # The linear forecasting head must be trained
                },
            )

        elif self.mode in ('LoRA', 'AdaLoRA'):
            config_dict = json.load(open('/hy-tmp/better464/MOMENT-1-large/config.json'))
            self.model = MOMENTPipeline_LORA(config=config_dict,
                                        model_kwargs={
                                            'task_name': 'forecasting',
                                            'forecast_horizon': self.forecast_horizon,
                                            'freeze_encoder': False,  # Freeze the patch embedding layer
                                            'freeze_embedder': False,  # Freeze the transformer encoder
                                            'freeze_head': False,  # The linear forecasting head must be trained
                                            'lora_mode': self.mode,
                                            'r': 16,
                                            'lora_alpha': 32,
                                            'target_modules': ["q", "v"],
                                            'lora_dropout': 0.03,
                                        })
            self.model.init()
            # 把参数加载到我们的模型中去
            load_weights(self.model)
            # 添加了这个部分，会导致部分反串梯度为None
            for n, p in self.model.named_parameters():
                # 设置这个为True 所有的都有梯度
                p.requires_grad = True

                # if 'lora_' not in n:
                #     p.requires_grad = False
                # else:
                #     p.requires_grad = True
                if not p.requires_grad:
                    print(n)
            # lora部分的参数是可修改的  这里需要修改
            # lora.mark_only_lora_as_trainable(self.model)  # 这个会导致 grad为None 加载的程序得重写
            # 测试
            # self.model.patch_embedding.value_embedding.weight.requires_grad = False
            # head部分修改为可修改的
            # mark_head_as_trainable(self.model)

        self.model.init()

        self.model.train()

        print(self.model)
        print('Model initialized, training mode: ', self.mode)
        print("Unfrozen parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print('    ', name)
        # using cross MSE loss for forecasting
        self.criterion = torch.nn.MSELoss()

        if self.mode in ('linear_probing','zero_shot', 'full_tuning', 'LoRA', 'AdaLoRA'):
            params = []
            for n, p in self.model.named_parameters():
                if 'lora_' in n:
                    print(n)
                    params += [{'params':[p]}]

            self.optimizer = torch.optim.Adam(params, lr=1e-4) # 只会更新优化器中的梯度
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()), lr=1e-4)
            # 方式2
            # self.optimizer = torch.optim.Adam(self.model.encoder.block[0].layer[0].parameters(), lr=1e-4)
            # Create a OneCycleLR scheduler
            max_lr = 1e-4
            total_steps = len(self.train_loader) * self.epochs
            self.scheduler = OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 初始化rankallocator
        if self.mode == 'AdaLoRA':
            self.rankallocator = RankAllocator(
                self.model, lora_r=12, target_rank=12,
                init_warmup=500, final_warmup=1500, mask_interval=100,
                total_step=3000, beta1=0.85, beta2=0.85,
            )

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            # self.log_file.write(f'Epoch {epoch+1}/{self.epochs}\n')
            self.epoch = epoch + 1

            #
            if self.mode == 'zero_shot':
                trues, preds, histories = self.evaluate_epoch()

            elif self.mode in ('linear_probing', 'full_tuning', 'LoRA'):
                self.train_epoch_lp()
                trues, preds, histories = self.evaluate_epoch()

            elif self.mode in ('AdaLoRA'):
                self.train_epoch_allocate_lp()
                trues, preds, histories = self.evaluate_epoch()

            else:
                raise ValueError(
                    'Invalid mode, please choose linear_probing, full_finetuning, or unsupervised_representation_learning')


    ####################################Training function##################################
    def train_epoch_lp(self):
        '''
        Train only forecasting head-linear_probing
        '''
        self.model.train()

        # Move the model to the GPU
        self.model = self.model.to(self.device)

        # Move the loss function to the GPU
        self.criterion = self.criterion.to(self.device)

        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Gradient clipping value
        max_norm = 5.0

        losses = []
        for timeseries, forecast, input_mask in tqdm(self.train_loader, total=len(self.train_loader)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(self.device)
            input_mask = input_mask.to(self.device)
            forecast = forecast.float().to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(timeseries, input_mask)

            loss = self.criterion(output.forecast, forecast)
            # 梯度清零
            self.optimizer.zero_grad(set_to_none=True)
            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            for name, parms in self.model.named_parameters():
                print(f'name:{name}, grad_requirs:{parms.requires_grad}')
                print(f'params:{parms}')
                print(f'grad_value:{parms.grad}')
                print(f'is leaf:{parms.is_leaf}\n')

            # Clip gradients
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            scaler.step(self.optimizer)
            scaler.update()
            # self.optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Train loss: {average_loss:.3f}")
        # Step the learning rate scheduler
        self.scheduler.step()


    def train_epoch_allocate_lp(self):
        '''
        Train with allocation
        '''
        self.model.train()

        # Move the model to the GPU
        self.model = self.model.to(self.device)

        # Move the loss function to the GPU
        self.criterion = self.criterion.to(self.device)

        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # 初始化rank allocator
        # rankallocator = RankAllocator(
        #     self.model, lora_r=12, target_rank=12,
        #     init_warmup=500, final_warmup=1500, mask_interval=100,
        #     total_step=3000, beta1=0.85, beta2=0.85,
        # )

        # Gradient clipping value
        max_norm = 5.0

        losses = []
        global_step = 0
        for timeseries, forecast, input_mask in tqdm(self.train_loader, total=len(self.train_loader)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(self.device)
            input_mask = input_mask.to(self.device)
            forecast = forecast.float().to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(timeseries, input_mask)

            loss = self.criterion(output.forecast, forecast)

            # 加上额外的约束
            # total_loss = loss + compute_orth_regu(self.model, regu_weight=0.1)
            # total_loss.backward()

            loss.backward()  # 观察一下 self.model.encoder.block[0].layer[0].SelfAttention.q
            for name, parms in self.model.named_parameters():
                print(f'name:{name}, grad_requirs:{parms.requires_grad}')
                print(f'grad_value:{parms.grad}')
            # self.rankallocator.update_and_mask(self.model, global_step)  # params grand

            self.optimizer.step()
            # # Scales the loss for mixed precision training
            # scaler.scale(loss).backward()
            #
            # # Clip gradients
            # scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            #
            # scaler.step(self.optimizer)
            # scaler.update()
            # # self.optimizer.zero_grad(set_to_none=True)

            # self.rankallocator.update_and_mask(self.model, global_step) # params grand

            self.optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            # 更新rankallocator
            # rankallocator.update_and_mask(self.model, global_step)
            global_step += 1

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Train loss: {average_loss:.3f}")
        # Step the learning rate scheduler
        # self.scheduler.step()


    ####################################Evaluate function##################################
    def evaluate_epoch(self):
        # Evaluate the model on the test split
        trues, preds, histories, losses = [], [], [], []
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for timeseries, forecast, input_mask in tqdm(self.test_loader, total=len(self.test_loader)):
                # Move the data to the GPU
                timeseries = timeseries.float().to(self.device)
                input_mask = input_mask.to(self.device)
                forecast = forecast.float().to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(timeseries, input_mask)

                loss = self.criterion(output.forecast, forecast)
                losses.append(loss.item())

                trues.append(forecast.detach().cpu().numpy())
                preds.append(output.forecast.detach().cpu().numpy())
                histories.append(timeseries.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        self.model.train()

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)

        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')

        print(f"Test Loss: {average_loss:.3f}| Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}")

        return trues, preds, histories




if __name__ == "__main__":
    # config_dict = json.load(open('/hy-tmp/better464/MOMENT-1-large/config.json'))
    # model = MOMENTPipeline_LORA(config=config_dict,
    #                        model_kwargs={
    #                            'task_name':'forecasting',
    #                            'forecast_horizon': 96,
    #                            'freeze_encoder': False,  # Freeze the patch embedding layer
    #                            'freeze_embedder': False,  # Freeze the transformer encoder
    #                            'freeze_head': False,  # The linear forecasting head must be trained
    #                            'lora_mode': 'AdaLoRA',
    #                            'r': 16,
    #                            'lora_alpha': 32,
    #                            'target_modules':["q", "v"],
    #                            'lora_dropout': 0.03,
    #                        })
    # model.init()
    # # # 把参数加载到我们的模型中去
    # # load_weights(model)
    # # # lora部分的参数是可修改的
    # # lora.mark_only_lora_as_trainable(model)
    # # # head部分修改为可修改的
    # # mark_head_as_trainable(model)
    # print(model)
    # total_params_info, params_list = get_pytorch_model_info(model)
    # # print(params_list)
    # ret_dict = filter_dic(params_list)

    #
    seed = 13
    control_randomness(seed)
    batch_size = 16
    epochs = 1
    output_path = '/hy-tmp/moment/tuning_exp/logs'
    # LP
    # mode = 'linear_probing'
    # Zeroshot
    # mode = 'zero_shot'
    # full_tuning
    # mode = 'full_tuning'
    # LoRA
    # mode = 'LoRA'
    # mode_list = ['linear_probing', 'zero_shot', 'full_tuning', 'LoRA']
    mode_list = ['LoRA']
    forecast_horizon_list = [96, 192, 336, 720]
    for mode in mode_list:
        for forecast_horizon in forecast_horizon_list:
            trainer = MOMENT_Trainer(seed, batch_size, epochs, forecast_horizon, mode, output_path)
            trainer.train()
            trainer.log_file.close()
            torch.cuda.empty_cache()
