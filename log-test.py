import logging

class ExampleLogger:
    def __init__(self, mode, imp_cal):
        self.log_file = os.path.join(self.log_path,
                                     f'log_{self.mode}_{self.forecast_horizon}_allnum:{self.allocate_num}'
                                     f'_importance_{self.importance_gate}_lorarank:{self.lora_rank}_lr:{self.max_lr}'
                                     f'_impcal_{self.imp_cal}_{timestamp}.log')
        self.logger = logging.getLogger(f'{mode}_{imp_cal}')
        self.logger.setLevel(logging.INFO)

        # 添加处理器
        handler = logging.StreamHandler()  # 这里使用控制台输出
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log(self, message):
        self.logger.info(message)

# 使用示例
logger1 = ExampleLogger('train', 'yes')
logger1.log("This is a training log message.")

logger2 = ExampleLogger('train', 'no')
logger2.log("This is a different log message.")