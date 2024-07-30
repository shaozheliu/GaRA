# 配置介绍

[toc]
本章主要提供一些必要的环境配置指南，包括代码环境配置、VSCODE 代码编辑器的 Python 环境配置，以及一些使用到的其他资源配置。

## 一、代码环境配置指南

这里我们详细介绍了代码环境配置的每一步骤，分为基础环境配置和通用环境配置两部分，以满足不同用户和环境的需求。

- **基础环境配置**部分：适用于环境配置**初学者**或**新的服务器环境（如阿里云）**。这部分介绍了如何生成 SSH key 并添加到 GitHub，以及在安装和初始化 conda 环境。

- **通用环境配置**部分：适用于**有一定经验的用户**、**已有环境基础**的本地安装或**完全独立的环境（如 GitHub Codespace）**。这部分介绍了如何新建和激活 conda 虚拟环境，克隆项目仓库，切换到项目目录，以及安装所需的 Python 包。为了加速 Python 包的安装，我们还提供了一些国内镜像源。*对于完全独立的环境，可以跳过前两步关于虚拟环境（conda）配置的步骤*。

### 1.1 基础环境配置(配置 git 和 conda)

1. 生成 ssh key
   `ssh-keygen -t rsa -C "youremail@example.com"`
2. 将公钥添加到 github
   `cat ~/.ssh/id_rsa.pub`
   复制输出内容，打开 github，点击右上角头像，选择 `settings` -> `SSH and GPG keys` -> `New SSH key`，将复制的内容粘贴到 key 中，点击 `Add SSH key`。
   

3. 安装 conda 环境

   1. linux 环境（通常采用 linux 环境）

      1. 安装：

         ```shell
         mkdir -p ~/miniconda3
         wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
         bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
         rm -rf ~/miniconda3/miniconda.sh
         ```

      2. 初始化：

         ```shell
         ~/miniconda3/bin/conda init bash
         ~/miniconda3/bin/conda init zsh
         ```

      3. 新建终端，检查 conda 是否安装成功 `conda --version`

   2. macOS 环境

      1. 安装

         ```shell
         mkdir -p ~/miniconda3
         curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
         bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
         rm -rf ~/miniconda3/miniconda.sh
         ```

      2. 初始化：

         ```shell
         ~/miniconda3/bin/conda init bash
         ~/miniconda3/bin/conda init zsh
         ```

      3. 新建终端，检查 conda 是否安装成功 `conda --version`
    
### 1.2 通用环境配置 

1. 新建虚拟环境
   `conda create -n llm-universe python=3.10`
   在这里指定你自己的大模型的名字
2. 激活虚拟环境
   `conda activate llm-universe`
3. cd到希望存储项目的路径下，并克隆当前仓库（github源码对应的仓库）不克隆也行
   `git clone git@github.com:datawhalechina/llm-universe.git`
4. 将目录切换到 llm-universe
   `cd llm-universe`
   ![cd_root](../../figures/C1-7-cd_root.png)
5. 安装所需的包
   `pip install -r requirements.txt`
   ![pip_install](../../figures/C1-7-pip_install.png)
   通常可以通过清华源加速安装
   `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
6. 如果pycharm debug出问题了，则删掉root环境下 rm -rv.pycharm_helpers  然后重新上传；
> 这里列出了常用的国内镜像源，镜像源不太稳定时，大家可以按需切换：
> 清华：https://pypi.tuna.tsinghua.edu.cn/simple/
> 阿里云：http://mirrors.aliyun.com/pypi/simple/
> 中国科技大学：https://pypi.mirrors.ustc.edu.cn/simple/
> 华中科技大学：http://pypi.hustunique.com/simple/
> 上海交通大学：https://mirror.sjtu.edu.cn/pypi/web/simple/
> 豆瓣：http://pypi.douban.com/simple

## 二、大模型下载指南

### 2.1 大模型参数下载到本地 

```python
import os
# 注意os.environ得在import huggingface库相关语句之前执行。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

def download_model(source_url, local_dir):
    # 使用huggingface原地址
    # source_url ="https://huggingface.co/BlinkDL/rwkv-4-novel/blob/main/RWKV-4-Novel-7B-v1-ChnEng-20230426-ctx8192.pth"
    # 使用huggingface-镜像地址
    # source_url = "https://hf-mirror.com/BlinkDL/rwkv-4-novel/blob/main/RWKV-4-Novel-7B-v1-ChnEng-20230426-ctx8192.pth"

    if 'blob' in source_url:
        sp = '/blob/main/'
    else:
        sp = '/resolve/main/'

    if 'huggingface.co' in source_url:
        url = 'https://huggingface.co/'
    else:
        url = 'https://hf-mirror.com'

    location = source_url.split(sp)
    repo_id = location[0].strip(url)  # 仓库ID，例如："BlinkDL/rwkv-4-world"

    cache_dir = local_dir + "/cache"


    filename = location[1]  # 大模型文件，例如："RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth"

    print(f'开始下载\n仓库：{repo_id}\n大模型：{filename}\n如超时不用管，会自定继续下载，直至完成。中途中断，再次运行将继续下载。')
    while True:
        try:
            hf_hub_download(cache_dir=cache_dir,
                        local_dir=local_dir,
                        repo_id=repo_id,
                        filename=filename,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        etag_timeout=100
                        )
        except Exception as e:
            print(e)
        else:
            print(f'下载完成，大模型保存在：{local_dir}\{filename}')
            break
if __name__ == '__main__':
    # 下载模型
    source_url = 'https://hf-mirror.com/AutonLab/MOMENT-1-large/blob/main/pytorch_model.bin'
    download_model(source_url, local_dir=r'/hy-tmp/MOMENT-1-large')
```

### 2.2 本地加载大模型


## 三、注意事项
1. 从modelscope上下载对应模型
2. 将模型保存到 /hy-tmp 文件夹下
3. 建立自己的conda环境并关联自己本地pycharm
4. 自己在服务器命令行上安装配置好对应的环境；
5. 使用服务器jupyter notebook的时候 pip install ipykernel 才能够用自己的环境运行代码
6. moment :cu118/torch-2.0.1%2Bcu118-cp311-cp311-linux_x86_64.whl
cu118/torch-2.0.1%2Bcu118-cp311-cp311-win_amd64.whl


### 虚拟环境
# 创建 Python 的虚拟环境，位置为 /hy-tmp/myenv
- conda create -p /hy-tmp/sz-moment python=3.11
- conda remove -p /hy-tmp/sz-moment --all
- conda activate /hy-tmp/sz-moment
- git clone https://github.com/moment-timeseries-foundation-model/moment.git
- pip install momentfm -i  https://pypi.mirrors.ustc.edu.cn/simple/
- pip install pandas -i  https://pypi.mirrors.ustc.edu.cn/simple/
- conda install ipykernel
- pip install scikit-learn -i  https://pypi.mirrors.ustc.edu.cn/simple/  注意要用pip，conda install会出毛病
- 如果要装peft，需要手动降级一些包的版本


# demo入口

## Moment 模型运行入口
- main.py 可直接运行
  - from momentfm_hand.data.informer_dataset import InformerDataset  需要修改路径："/hy-tmp/data/ETT-small/ETTh1.csv" 为自己的ETTh1.csv绝对路径
```python
# self.model.init()
# 把参数加载到我们的模型中去,加入后可正常训练
load_weights(self.model)
# 将除了lora部分的参数requires_grad=False
mark_only_lora_as_trainable(self.model)  # 这个会导致 grad为None 加载的程序得重写
```
- 修改后的model：MOMENTPipeline_LORA 
```python
self.encoder = self._get_transformer_backbone(config)
```
- lora添加的位置：modeling_t5_lora.py- T5Attention
- lora部分函数 lora_tuils

## Moment Peft测试demo
- peft_tutorials/finetune_demo/forecasting.py  可正常运行，需要修改InformerDataset路径