#模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('better464/MOMENT-1-large', cache_dir='/hy-tmp/MOMENT-1-large')
from momentfm import MOMENTPipeline





from transformers import AutoModel

# 假设您的模型权重文件(.bin文件或.pth文件)位于当前目录的'model_directory'文件夹中
# 如果模型权重文件具有不同的文件扩展名，请确保指定正确的文件扩展名
# 例如，对于PyTorch模型，使用'pytorch_model.bin'

# 加载模型
model = AutoModel.from_pretrained("MOMENT-1-large", cache_dir="/hy-tmp/MOMENT-1-large/better464/MOMENT-1-large")

# 现在您可以使用加载的模型进行推理了