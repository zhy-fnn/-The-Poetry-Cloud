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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2.必要文件

请确保当前文件夹下存在如下文件：

| 路径 | 用途 |
|---|---|
| data/poetry_char_v3/meta.pkl | 字符表 |
| data/poetry_char_v3/yun_trad.json | 押韵检查 |
| data/poetry_sft_v1/meta.pkl | SFT 数据元信息 |
| data/poetry_sft_v1/train.bin | SFT 训练数据 |
| data/poetry_sft_v1/val.bin | SFT 验证数据 |
| data/poetry_sft_v1/sft_manifest.json | 数据统计 |
| out/pretrain/ckpt/ckptbest.pt | 预训练检查点 |

### 3.验证安装

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
[dry-run] data: E:\apeiria\ML project\the poetry cloud\data\poetry_sft_v1
[dry-run] out: E:\apeiria\ML project\the poetry cloud\out\sft
[dry-run] vocab: 8888
[dry-run] train: 19801 首 / 1207124 tokens
[dry-run] val: 199 首 / 11944 tokens
[dry-run] init ckpt: E:\apeiria\ML project\the poetry cloud\out\pretrain\ckpt\ckptbest.pt
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
[dry-run] data: E:\apeiria\ML project\the poetry cloud\data\poetry_sft_v1
[dry-run] out: E:\apeiria\ML project\the poetry cloud\out\sft
[dry-run] vocab: 8888
[dry-run] train: 19801 首 / 1207124 tokens
[dry-run] val: 199 首 / 11944 tokens
[dry-run] init ckpt: E:\apeiria\ML project\the poetry cloud\out\pretrain\ckpt\ckptbest.pt
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
| sample | 从采样点生成古诗 |
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

