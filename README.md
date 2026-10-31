# the poetry cloud 诗云

> 这是基于字符集GPT做出的中文古诗生成项目，项目的源头是nanoGPT，我在此做出了一些修改

> 当前状态：v0.1，CLI.py 的train，sample，best_of_n均可运行

## 当前能力

根据提示词和诗句类型生成绝句或律诗

从预训练模型生成的多首诗里面筛选出符合格律的诗

预训练模型和SFT模型生成诗的对照

## 快速开始

### 1.环境要求

请确保环境中有如下配置：

- Python 3.12 - 3.14

- Windows / Linux

- CPU 可运行，有CUDA时自动使用GPU加速

Windows 安装：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux / macOS 安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2.必要文件

请确保当前文件夹下存在如下文件：

| 路径 | 用途 |
|---|---|
| data/poetry_char_v2/meta.pkl | 分词器字符表 |
| data/poetry_char_v3/meta.pkl | 采样字符表 |
| data/poetry_char_v3/yun_trad.json | 押韵检查 |
| data/poetry_sft_v1/meta.pkl | SFT 数据元信息 |
| data/poetry_sft_v1/train.bin | SFT 训练数据 |
| data/poetry_sft_v1/val.bin | SFT 验证数据 |
| data/poetry_sft_v1/sft_manifest.json | 数据统计 |
| out/pretrain/ckpt/ckptbest.pt | 预训练检查点 |

### 3.下载运行数据

仓库不包含 `data/`。请下载最小运行数据包，并解压到项目根目录：

- [poetry-cloud-runtime-data-v0.1.0.zip](https://github.com/zhy-fnn/-The-Poetry-Cloud/releases/download/v0.1.0/poetry-cloud-runtime-data-v0.1.0.zip)
- 文件大小：1.671 MB
- SHA256：`26DF481DA5ED3DCDD4DD69C18A17842307171AA3799C2C60654D785F0AF5A237`

```powershell
Invoke-WebRequest `
  -Uri https://github.com/zhy-fnn/-The-Poetry-Cloud/releases/download/v0.1.0/poetry-cloud-runtime-data-v0.1.0.zip `
  -OutFile poetry-cloud-runtime-data-v0.1.0.zip

Get-FileHash .\poetry-cloud-runtime-data-v0.1.0.zip -Algorithm SHA256
Expand-Archive .\poetry-cloud-runtime-data-v0.1.0.zip -DestinationPath .
```

### 4.下载模型检查点

仓库不包含模型权重，请从 GitHub Release 下载：

- [ckptbest.pt](https://github.com/zhy-fnn/-The-Poetry-Cloud/releases/download/v0.1.0/ckptbest.pt)
- 文件大小：269.58 MB
- SHA256：`7B4AFF1FBB0C9A85C9EB769CF26C793C8C2C30725E313502AF5995353315ED81`

Windows PowerShell 中创建目录并检查文件：

```powershell
New-Item -ItemType Directory -Force out\pretrain\ckpt
Get-FileHash .\out\pretrain\ckpt\ckptbest.pt -Algorithm SHA256
```

将下载的 `ckptbest.pt` 放到 `out/pretrain/ckpt/ckptbest.pt`。文件哈希应与上面的 SHA256 一致。

### 5.验证安装

```powershell
python CLI.py --help
python CLI.py train --dry-run
```

期待结果：

```text
诗云的命令行入口

positional arguments:
  {sample,train,best_of_n}
    sample              用ckpt生成诗
    train               打印训练配置
    best_of_n           生成n个诗中最好的

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
[dry-run] data: .\the poetry cloud\data\poetry_sft_v1
[dry-run] out: .\the poetry cloud\out\sft
[dry-run] vocab: 8888
[dry-run] train: 19801 首 / 1207124 tokens
[dry-run] val: 199 首 / 11944 tokens
[dry-run] init ckpt: .\the poetry cloud\out\pretrain\ckpt\ckptbest.pt
[dry-run] all passed
```

## 命令用法

### 生成古诗

```powershell
python CLI.py sample `
  --ckpt out/pretrain/ckpt/ckptbest.pt `
  --prompt 溫泉 `
  --genre 绝句 `
  --count 1
```

期待结果：

```text
--- 溫泉 第 1 首 ---
一般寒月綠花深，萬里靈聲未復風。
莫是西西天影去，滿雲風在水陰中。
```

### 检查训练数据

```powershell
python CLI.py train --dry-run
```

期待结果：

```text
[dry-run] data: .\the poetry cloud\data\poetry_sft_v1
[dry-run] out: .\the poetry cloud\out\sft
[dry-run] vocab: 8888
[dry-run] train: 19801 首 / 1207124 tokens
[dry-run] val: 199 首 / 11944 tokens
[dry-run] init ckpt: .\the poetry cloud\out\pretrain\ckpt\ckptbest.pt
[dry-run] all passed
```

### 生成best of n

冒烟测试

```powershell
python CLI.py best_of_n --smoke --out_dir out/best_of_n_smoke
```

期待结果：

```text
溫泉 2000 True False
```

正式生成

```powershell
python CLI.py best_of_n --n_per_titles 30
```

期待结果：

```text
溫泉 2000 True False
溫泉 2001 True True
溫泉 2002 True False
溫泉 2003 True False
溫泉 2004 True False
溫泉 2005 True False
溫泉 2006 True False
溫泉 2007 True False
溫泉 2008 True False
溫泉 2009 True True
溫泉 2010 True False
溫泉 2011 True True
溫泉 2012 True False
溫泉 2013 True True
溫泉 2014 True False
………………
```

| 命令 | 作用 |
| --- | --- |
| sample | 从检查点生成古诗 |
| train --dry-run | 测试训练数据是否正确 |
| best_of_n | 生成多首诗并选出符合韵律的 |
| best_of_n --smoke | 小范围测试 |

## 项目结构

这里只列出核心结构

```text
the poetry cloud/
├─ CLI.py              # 命令行入口
├─ sample.py           # 单次生成
├─ best_of_N.py        # 多候选生成与筛选
├─ model.py            # GPT 模型
├─ tokenizer.py        # 字符级分词
├─ rhythm.py           # 格律与押韵检查
├─ train.py            # 预训练
├─ train_sft.py        # SFT 训练
├─ eval_compare.py     # 模型评测
├─ data/               # 数据
└─ out/                # 检查点与实验产物
```

## 模型与参数

简要写清楚影响复现的信息：

·数据版本：poetry_char_v3

·训练样本：约 22.6 万首

·字符表大小：8888

·预训练最佳步数：10500

·验证集 loss：4.4816

·SFT 最佳步数：500

·SFT 验证集 loss：4.2399