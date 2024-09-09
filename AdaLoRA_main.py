import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from momentfm_hand.models.moment_LoRA import MOMENT_LORA, MOMENTPipeline_LORA
from momentfm import MOMENTPipeline
import json
import torch
import torch.nn.init as init
import loralib as lora
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from itertools import islice
from momentfm_hand.data.informer_dataset import InformerDataset
from momentfm.utils.utils import control_randomness
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from peft import get_peft_model, TaskType, AdaLoraConfig



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
    def __init__(self, seed, batch_size, epochs, forecast_horizon, mode, log_path, checkpoint_path):
        # initialize ptbxl classification dataset
        self.mode = mode
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.train_dataset = InformerDataset(data_split="train", random_seed=seed,
                                             forecast_horizon=self.forecast_horizon)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataset = InformerDataset(data_split="val", random_seed=seed,
                                             forecast_horizon=self.forecast_horizon)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = InformerDataset(data_split="test", random_seed=seed, forecast_horizon=self.forecast_horizon)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.importance_gate = 0.1
        self.allocate_num = 4
        # create log file to store training logs
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # create log file to store training logs
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.log_file = open(os.path.join(self.log_path, f'log_{self.mode}_{self.forecast_horizon}_importance_{self.importance_gate}.txt'), 'w')
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
            for n, p in self.model.named_parameters():
                # 设置这个为True 所有的都有梯度
                p.requires_grad = True

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

        elif self.mode in ('LoRA'):
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
                                            'lora_dropout': 0.1,
                                        })
            self.model.init()
            # 把参数加载到我们的模型中去
            load_weights(self.model)

            # lora部分的代码，由于weight部分requires grad = False，会导致前面层梯度为None
            for n, p in self.model.named_parameters():
                # 设置这个为True 所有的都有梯度
                p.requires_grad = True

        elif self.mode in ('AdaLoRA'):
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
            self.model.init()
            lora_config = AdaLoraConfig(
                init_r=16,
                target_r=10,
                tinit=50,
                tfinal=100,
                deltaT=5,
                beta1=0.3,
                beta2=0.3,
                orth_reg_weight=0.2,
                lora_alpha=32,
                target_modules=["q", 'k', 'v'],
                lora_dropout=0.1,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            print('AdaLoRA enabled')
            self.model.print_trainable_parameters()


        self.model.init()
        print(self.model)
        # 打印可训练参数
        print(self.model.print_trainable_parameters())
        # using cross MSE loss for forecasting
        self.criterion = torch.nn.MSELoss()

        if self.mode in ('linear_probing','zero_shot', 'full_tuning', 'LoRA'):
            params = []
            for n, p in self.model.named_parameters():
                if 'lora_' in n:
                    params += [{'params':[p]}]
                if 'head' in n:
                    params += [{'params':[p]}]
            # 将params加入optimizer，仅更新这一部分
            self.optimizer = torch.optim.Adam(params, lr=1e-4)
        if self.mode in ('AdaLoRA'):
            self.optimizer = torch.optim.Adam(self.model.named_parameters(), lr=1e-4)
            max_lr = 1e-4
            total_steps = len(self.train_loader) * self.epochs
            self.scheduler = OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(self):
        # 初始化
        self.model.train()

        # Move the model to the GPU
        self.model = self.model.to(self.device)

        # Move the loss function to the GPU
        self.criterion = self.criterion.to(self.device)
        min_loss = 999
        for epoch in tqdm(range(self.epochs)):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            self.epoch = epoch + 1

            if self.mode == 'zero_shot':
                trues, preds, histories = self.evaluate_epoch()

            elif self.mode in ('linear_probing', 'full_tuning', 'LoRA', 'AdaLoRA'):
                self.train_epoch_lp()
                checkpoint_path = os.path.join(self.checkpoint_path, f'mode:{self.mode}_forecast:{self.forecast_horizon}.pth')
                print('Start pred')
                trues, preds, histories, average_loss = self.evaluate_epoch()
                if average_loss < min_loss:
                    min_loss = average_loss
                    torch.save(lora.lora_state_dict(self.model), checkpoint_path)
                    self.plot_show(trues, preds, histories, average_loss)

            elif self.mode in ('AdaLoRA'):
                self.train_epoch_allocate_lp()
                # 拼接文件路径
                checkpoint_path = os.path.join(self.checkpoint_path, f'mode:{self.mode}_forecast:{self.forecast_horizon}.pth')
                print('Start pred')
                trues, preds, histories, average_loss = self.evaluate_epoch()
                if average_loss < min_loss:
                    min_loss = average_loss
                    torch.save(lora.lora_state_dict(self.model), checkpoint_path)
                    print('Start pred')
                    self.plot_show(trues, preds, histories, average_loss)

            else:
                raise ValueError(
                    'Invalid mode, please choose linear_probing, full_finetuning, or unsupervised_representation_learning')



    ####################################Training function##################################
    def train_epoch_lp(self):
        '''
        Train only forecasting head-linear_probing
        '''
        scaler = torch.cuda.amp.GradScaler()

        # Gradient clipping value
        max_norm = 5.0

        losses = []
        for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
        # for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(islice(self.train_loader, 4))):
            # Move the data to the GPU
            timeseries = timeseries.float().to(self.device)
            input_mask = input_mask.to(self.device)
            forecast = forecast.float().to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(timeseries, input_mask)

            loss = self.criterion(output.forecast, forecast)

            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            scaler.step(self.optimizer)
            scaler.update()
            # 梯度清零
            self.optimizer.zero_grad()


            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Train loss: {average_loss:.3f}")
        # Step the learning rate scheduler
        self.scheduler.step()


    def train_epoch_allocate_lp(self):
        '''
        Train only forecasting head-linear_probing
        '''
        scaler = torch.cuda.amp.GradScaler()

        # Gradient clipping value
        max_norm = 5.0

        losses = []
        for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
        # for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(islice(self.train_loader, 4))):
            # Move the data to the GPU
            timeseries = timeseries.float().to(self.device)
            input_mask = input_mask.to(self.device)
            forecast = forecast.float().to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(timeseries, input_mask)

            loss = self.criterion(output.forecast, forecast)

            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            scaler.step(self.optimizer)
            scaler.update()
            # 梯度清零
            self.optimizer.zero_grad()

            if batch_idx % (len(self.train_loader) // self.allocate_num) == 0:
                print("=" * 60)
                print(f'Start allocating rank {batch_idx//self.allocate_num}')
                lora_names, neuron_importance, neuron_avg_score = self.compute_neuron_lora_importance()
                self.mask_lora_neuron(lora_names, neuron_importance, neuron_avg_score, self.importance_gate)



            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Train loss: {average_loss:.3f}")
        # Step the learning rate scheduler
        self.scheduler.step()


    ####################################Eval grads#########################################

    def compute_neuron_gate_importance(self):
        """ This method shows how to compute:
            - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
        """

        # self.model.train()
        #
        # # Move the model to the GPU
        # self.model = self.model.to(self.device)
        #
        # # Move the loss function to the GPU
        # self.criterion = self.criterion.to(self.device)
        # 初始化
        lora_names = []
        lora_weights = []
        for name, params in self.model.named_parameters():
            if 'lora_E' in name:
                if params.dim() > 1:
                    lora_names.append(name)
                    lora_weights.append(params)
                else:
                    pass

        # 初始化神经元分数重要性
        neuron_importance = []
        for w in lora_weights:
            neuron_importance.append(torch.zeros(w.shape[0]).to(self.device))

        for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(self.val_loader, total=len(self.val_loader))):
        # for batch_idx, (timeseries, forecast, input_mask) in enumerate(islice(self.train_loader, 2)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(self.device)
            input_mask = input_mask.to(self.device)
            forecast = forecast.float().to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self.model(timeseries, input_mask)

            loss = self.criterion(output.forecast, forecast)

            loss.backward()  # 观察一下 self.model.encoder.block[0].layer[0].SelfAttention.q
            # 计算LoRA_E每一个batch上的贡献度
            for name, weight, current_importance in zip(lora_names, lora_weights, neuron_importance):
                current_importance += ((weight * weight.grad).sum(dim=1)).abs().detach()

        # 计算平均重要性
        neuron_importance_avg = [imp / len(self.val_loader)  for imp in neuron_importance]
        return lora_names, neuron_importance_avg

    def compute_neuron_lora_importance(self):
        """ This method shows how to compute:
            - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
        """


        lora_E_names = []
        lora_E_weights = []
        lora_A_names = []
        lora_A_weights = []
        lora_B_names = []
        lora_B_weights = []
        for name, params in self.model.named_parameters():
            if 'lora_E' in name:
                if params.dim() > 1:
                    lora_E_names.append(name)
                    lora_E_weights.append(params)
                else:
                    pass
            if 'lora_A' in name:
                if params.dim() > 1:
                    lora_A_names.append(name)
                    lora_A_weights.append(params)
                else:
                    pass
            if 'lora_B' in name:
                if params.dim() > 1:
                    lora_B_names.append(name)
                    lora_B_weights.append(params)
                else:
                    pass


        # 初始化gate神经元重要性参数
        neuron_importance = []
        for w in lora_E_weights:
            neuron_importance.append(torch.zeros(w.shape[0]).to(self.device))

        # for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(self.val_loader, total=len(self.val_loader))):
        for batch_idx, (timeseries, forecast, input_mask) in enumerate(islice(self.train_loader, 2)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(self.device)
            input_mask = input_mask.to(self.device)
            forecast = forecast.float().to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self.model(timeseries, input_mask)

            loss = self.criterion(output.forecast, forecast)

            loss.backward()  # 观察一下 self.model.encoder.block[0].layer[0].SelfAttention.q
            # 计算LoRA_E每一个batch上的贡献度
            # for name, weight, current_importance in zip(lora_names, lora_weights, neuron_importance):
            for A_weight, B_weight, current_importance in zip(lora_A_weights, lora_B_weights, neuron_importance):
                current_importance += ((A_weight * A_weight.grad).sum(dim=1)).abs().detach()
                current_importance += ((B_weight * B_weight.grad).sum(dim=0)).abs().detach()

        # 计算平均重要性
        neuron_importance_avg = [imp / len(self.val_loader) for imp in neuron_importance]
        neuron_avg_score = torch.mean(torch.stack(neuron_importance_avg), dim=0)
        return lora_E_names, neuron_importance_avg, neuron_avg_score

    def mask_lora_neuron(self, lora_names, neuron_importance, neuron_avg_score, importance_gate):
        """ reorder neurons based on their importance.

            Arguments:
                model: bert model
                head_importance: 12*12 matrix for head importance in 12 layers
                neuron_importance: list for neuron importance in 12 layers.
        """
        for name, params in self.model.named_parameters():
            if 'lora_E' in name:
                # Get the index of the parameter name in lora_names
                lora_index = lora_names.index(name)
                # Get the corresponding importance score
                importance_scores = neuron_importance[lora_index]
                # print(importance_scores)
                gate_value = neuron_avg_score * importance_gate
                boolen_mask = importance_scores  < gate_value
                with torch.no_grad():  # 禁用梯度计算
                    # 将模型参数中对应为 True 的位置置为 0
                    params[boolen_mask] = 0
                    print(name)
                    print(f'New:{params}')



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

        return trues, preds, histories, average_loss

    def plot_show(self, trues, preds, histories, average_loss):
        # Assuming histories, trues, and preds are your lists containing the data
        # Extracting the first data point

        channel_idx = np.random.randint(0, 7)  # There are 7 channels in this dataset
        time_index = np.random.randint(0, trues.shape[0])

        history = histories[time_index, channel_idx, :]
        true = trues[time_index, channel_idx, :]
        pred = preds[time_index, channel_idx, :]

        plt.figure(figsize=(12, 4))

        # Plotting the first time series from history
        plt.plot(range(len(history)), history, label=f'History (512 timesteps)', c='darkblue')

        # Plotting ground truth and prediction
        num_forecasts = len(true)

        offset = len(history)
        plt.plot(range(offset, offset + len(true)), true, label='Ground Truth', color='darkblue',
                 linestyle='--', alpha=0.5)
        plt.plot(range(offset, offset + len(pred)), pred, label='Forecast', color='red', linestyle='--')

        plt.title(f"ETTh1 (Hourly) -- (idx={time_index}, channel={channel_idx}), "
                  f"epoch={self.epoch}, loss= {average_loss}", fontsize=18)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(fontsize=14)
        # 设置DPI为300来增加分辨率
        plt.savefig(f'{self.log_path}/test_res_{self.mode}_{self.forecast_horizon}_gateimp:{self.importance_gate}.png', dpi=300)


if __name__ == "__main__":

    seed = 13
    control_randomness(seed)
    batch_size = 32
    epochs = 20

    mode_list = ['AdaLoRA']
    # forecast_horizon_list = [96, 192, 336, 720]
    forecast_horizon_list = [96]
    for mode in mode_list:
        log_path = f'./logs/{mode}'
        checkpoint_path = f'./checkpoints/{mode}'
        for forecast_horizon in forecast_horizon_list:
            trainer = MOMENT_Trainer(seed, batch_size, epochs, forecast_horizon, mode, log_path, checkpoint_path)
            trainer.train()
            trainer.log_file.close()
            torch.cuda.empty_cache()
