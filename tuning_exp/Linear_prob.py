import torch
from momentfm import MOMENTPipeline
import random
from functools import partial
import os
import sys
import numpy as np
# import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import argparse
from functools import partial
from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics


def extract_qkv_layers(model, qkv_flg, parent_name=''):
    qkv_layers = []
    for name, module in model.named_children():
        module_name = f"{parent_name}.{name}" if parent_name else name

        if qkv_flg and hasattr(module, 'q') and hasattr(module, 'k') and hasattr(module, 'v'):
            qkv_layers.append(module)

        qkv_layers_child = extract_qkv_layers(module, qkv_flg, parent_name=module_name)
        qkv_layers.extend(qkv_layers_child)

    return qkv_layers


def LoRA_init(model, qkv_flg):
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    qkv_layers = extract_qkv_layers(model, qkv_flg)
    for layer in qkv_layers:
        if qkv_flg == True:
            layer.q = assign_lora(layer.q)
            layer.k = assign_lora(layer.k)
            layer.v = assign_lora(layer.v)
    print('Finish LoRA init')
    print(model)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.W_a @ self.W_b)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)





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

        elif self.mode in ('LoRA'):
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
            LoRA_init(self.model, True)


        self.model.init()
        print('Model initialized, training mode: ', self.mode)
        # using cross MSE loss for forecasting
        self.criterion = torch.nn.MSELoss()

        if self.mode in ('linear_probing','zero_shot', 'full_tuning', 'LoRA'):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            # Create a OneCycleLR scheduler
            max_lr = 1e-4
            total_steps = len(self.train_loader) * self.epochs
            self.scheduler = OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Train loss: {average_loss:.3f}")
        # Step the learning rate scheduler
        self.scheduler.step()

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

if __name__ == '__main__':
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
    mode_list = ['linear_probing']
    forecast_horizon_list = [96, 192, 336, 720]
    for mode in mode_list:
        for forecast_horizon in forecast_horizon_list:
            trainer = MOMENT_Trainer(seed, batch_size, epochs, forecast_horizon, mode, output_path)
            trainer.train()
            trainer.log_file.close()
            torch.cuda.empty_cache()