# 中文 Dense LLM 预训练

数据源为 `datasets/train.txt`（约 12.7 GB）。假设 UTF-8、每个非空行一篇文档；请先确认实际数据符合此约定。原始文件不会被修改。

## 快速开始（在本项目目录运行）

```powershell
pip install -r requirements.txt
python prepare.py --input datasets/train.txt --output data
python train.py --data data --out runs/base
python generate.py --checkpoint runs/base/best.pt --prompt "人工智能的发展"
```

GPU 训练请安装与驱动适配的 CUDA 版 PyTorch。默认自动选择 CUDA，否则 CPU（仅建议用于小规模验证）。未提供多卡分布式训练。显存不足时先减小 `--batch-size` 或 `--seq-len`，再增大 `--accum` 保持有效 batch。

## 模型与训练

约 102.45M 参数（词表实际大小可能略小，最终以 config.json 为准）：12 层、隐藏维度 768、12 个注意力头、SwiGLU 中间维度 1792、共享输入/输出 embedding、RMSNorm、RoPE、因果 SDPA 注意力。Dense 架构，无 MoE。上下文默认 1024。使用 AdamW、warmup + cosine 学习率、梯度累积、梯度裁剪、CUDA BF16/FP16 自动混合精度。词表损失分块并 checkpoint，以减少 logits 显存占用。

默认 10000 步，每步 32768 tokens，总计 327,680,000 tokens。这是可调整的运行预算，不代表充分预训练；100M 模型可将约 2B tokens 作为初步实验预算，再依据验证集 loss 决定是否延长。例如：

```powershell
python train.py --data data --out runs/2b --steps 61036 --batch-size 2 --seq-len 1024 --accum 16
```

训练随机抽取连续 token 窗口（有放回），不按 epoch 遍历。`tokens_seen` 包含重复采样，不等于唯一语料覆盖量。EOS 分隔文档，窗口允许跨文档注意力。

## BBPE 与二进制数据

Hugging Face Tokenizers 的 ByteLevel + BPE，完整 256-byte alphabet，词表目标 32000，无文本归一化，支持任意 UTF-8 文本的字节表示。特殊 token 为 pad/unk/bos/eos，每篇文档结尾追加 EOS。

默认用前一百万行中的训练集文本拟合 tokenizer（控制内存与耗时，可能受语料排序影响），随后流式分词全部文本。可设置 `--tokenizer-lines 0` 使用全部训练文本拟合，但耗时及内存会显著增长。`--limit-lines` 用于小样本测试。

按文本 BLAKE2 哈希划分 99% train / 1% val；相同的 strip 后文本不会跨集合，但未做近似去重。验证集不参与 tokenizer 拟合。

- `data/tokenizer.json`：可加载的 BBPE tokenizer。
- `data/train.bin` / `data/val.bin`：无文件头、连续 little-endian uint16 token ID，使用 numpy.memmap 读取。
- `data/meta.json`：dtype、token 数、文档数、实际词表、tokenizer SHA256、预处理参数。

预处理需额外磁盘空间，二进制大小为 token 总数乘以 2 字节。输出目录已有 meta.json 时拒绝覆盖；中断预处理后请使用新输出目录。

## 指标与整理

每次验证保存 `metrics.csv` 和 `loss_and_metrics.png`。CSV 每步记录训练 loss、验证 loss/PPL（仅验证步）、学习率、裁剪前梯度范数、tokens/s、单步耗时、累计 tokens、CUDA 峰值 allocated 显存。吞吐量不含验证、绘图及保存时间；显存包含当步验证。验证固定随机窗口，属于采样估计，不是全验证集精确 PPL；PPL 指数最大截断到 80。

`config.json` 保存模型/训练参数、软件版本及数据元信息；`summary.json` 保存最终概要。`best.pt` 按最低验证 loss 保存，`last.pt` 周期保存；包括模型、优化器、scaler、随机状态和历史指标。只加载可信 checkpoint。

```powershell
python train.py --data data --out runs/base --resume runs/base/last.pt
```

续训须保持原 steps、batch-size、seq-len、accum、warmup、lr、seed 和模型配置；自定义运行需重复传入原参数，以保持学习率日程一致。中断后从最近 checkpoint 恢复，之后的未保存步骤会重算。暂不支持改变训练日程的续训。

loss/PPL 衡量下一 token 预测，不能代替中文知识、推理或下游任务评测。本项目输出可用于实验对比，未内置这些能力基准。

## 小规模验证

```powershell
python prepare.py --output smoke_data --limit-lines 300 --tokenizer-lines 300 --vocab-size 512 --val-ratio .2
python train.py --data smoke_data --out runs/smoke --tiny --steps 2 --batch-size 1 --seq-len 32 --accum 1 --eval-every 1 --eval-batches 2 --warmup 1
```

`--tiny` 仅用于验证管道，不是 100M 正式模型。正式训练不传此参数。

参考：[BBPE trainer](https://huggingface.co/docs/tokenizers/main/en/api/trainers)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## SFT：distill_r1_110k_sft.jsonl

使用实际字段 `instruction`（指令）、`input`（可选补充输入）、`output`（蒸馏回答）。问题由 instruction 和非空 input 合并；repo_name、score 和原始 token 长度等元信息不进入模型。原始 token 长度来自其他分词器，不用于本模型截断。

先完成预训练，然后执行：

```powershell
python prepare_sft.py --input datasets/distill_r1_110k_sft.jsonl --tokenizer data/tokenizer.json --output sft_data --seq-len 1024
python sft.py --data sft_data --out runs/sft --init-from runs/base/best.pt
```

必须复用预训练 tokenizer，禁止重新训练词表。默认完整监督 output，包括其中已有的 `<think>...</think>` 和最终回答，不重复添加思考标记。若只希望学习最终回答，预处理时添加 `--answer-only`，删除开头的 think 段；标记未闭合或删除后为空时跳过并记录到 rejected.jsonl。思考标记作为普通文本编码，不扩充词表。

模板为 `User:\n{instruction + 可选 input}\n\nAssistant:\n{回答}<eos>`。回答和 EOS 参与损失，问题、角色前缀和 padding 的 labels 为 -100。按问题哈希划分 95%/5%，相同问题不跨集合。每条记录不跨样本拼接；长样本按 seq-len 切块，相邻块重叠一个 token，避免遗漏边界预测；续块只有当前块上下文，可能缺少原始问题，可增大 seq-len 缓解。纯问题块不会进入训练。

`train.bin` / `val.bin` 为每条 seq-len+1 个 uint16 ID 的定长记录，`train.mask.bin` / `val.mask.bin` 为对应 uint8 监督掩码；训练加载时右移标签，生成 -100。padding 在右侧，因果注意力保证它不影响前面的有效位置。meta.json 保存记录数、原始样本数、监督 token 数、文件哈希等。训练会验证文件哈希。

SFT 默认 2000 步、lr=2e-5、warmup=100，其他参数与 train.py 相同。不是按 epoch 遍历，按记录有放回采样，长回答的块数更多，所占权重也更多。每个 microbatch 的 loss 按有效回答 token 平均，再对梯度累积的 microbatch 等权平均；验证 loss 按有效 token 加权。SFT PPL 仅针对回答。tokens_seen/tokens_per_sec 为输入位置数（包含 padding），不代表监督 token 数。输出与预训练一致：进度条、CSV、loss/指标图、config.json、summary.json、best.pt、last.pt。

### 两种训练的断点续训

`--init-from` 仅用于首次 SFT，加载模型权重并创建新优化器；`--resume` 恢复完整训练状态，两者不能同时传入。

```powershell
# 预训练恢复（自定义参数需与首次运行一致）
python train.py --data data --out runs/base --resume runs/base/last.pt
# SFT 恢复，无需再传 init-from
python sft.py --data sft_data --out runs/sft --resume runs/sft/last.pt
```

每 `--save-every` 步（默认 500）及训练结束写入 last.pt，验证改善时写入 best.pt；先写临时文件再替换。恢复模型、优化器、混合精度 scaler、步数、最佳分数、历史指标、NumPy/PyTorch/CUDA 随机状态，学习率按原总步数恢复。相同环境中可重现随机采样；跨设备和非确定性 CUDA 内核不保证逐位一致。

要主动暂停并保存，可首次设置完整日程，再使用 `--stop-after`：

```powershell
python sft.py --init-from runs/base/best.pt --steps 2000 --stop-after 100
python sft.py --steps 2000 --resume runs/sft/last.pt
```

意外终止或 Ctrl+C 后从最近已完成的检查点恢复，未保存步骤重算，不保存半个梯度累积步骤。`--stop-after` 是绝对训练步数，不修改总训练日程。已完成的 checkpoint 可再次加载，程序会正常退出。修改 steps 等日程参数会被拒绝，避免无意改变续训语义。

SFT 推理自动使用同一聊天模板：

```powershell
python generate.py --checkpoint runs/sft/best.pt --tokenizer sft_data/tokenizer.json --prompt "什么是人工智能？"
```

运行 `python test_training.py` 会从本地真实语料自动创建小样本和临时目录，检查预训练及 SFT 连续训练/断点恢复权重逐位一致、masked loss 和 R1 字段解析；不会覆盖已有运行目录。


正式训练尚未在全量数据上运行；先使用小样本验证配置和显存。所有数据集、bin、checkpoint、运行产物均由 .gitignore 排除，不上传 GitHub。

已对本地 R1 文件进行全量字段检查：110000 条中 109999 条有效，1 条 instruction 为空；109709 条有效问题包含汉字。预处理将无效记录的行号及原因写入 rejected.jsonl，并在 meta.json 记录数量。该统计不代表内容质量或准确性评测。

## Linux 云 GPU 一键启动

选择 **Python 3.11+、已安装 CUDA 版 PyTorch 2.4+ 的 NVIDIA GPU 镜像**。脚本使用当前环境；不自动安装显卡驱动或更换 CUDA。若镜像不包含可用 PyTorch，先按 [官方安装选择器](https://pytorch.org/get-started/locally/) 安装匹配版本。

```bash
git clone https://github.com/adrian-mo61/nanollm.git
cd nanollm
# 将 train.txt 和 distill_r1_110k_sft.jsonl 上传到 datasets/
bash run.sh setup                  # 检查 Python/CUDA，安装项目依赖
bash run.sh all                    # 分词 → 预训练 → SFT 分词 → SFT
```

只跑某个阶段：`bash run.sh pretrain` 或 `bash run.sh sft`。默认输出在 cloud_work/，终端日志在 logs/。重复运行同一命令自动复用完成的预处理，并恢复 last.pt（不存在时回退 best.pt）。续训从 checkpoint 读取原始 batch、上下文和学习率日程，新传入的这些参数不覆盖原设置；已完成阶段加载检查后直接结束。没有检查点时无法恢复未保存的步骤。预处理中断不会自动续接，脚本会提示使用新工作目录或移走不完整的数据目录，避免误用半成品。

云服务器建议将数据和工作目录放在持久化磁盘：

```bash
bash run.sh all --work-dir /mnt/data/nanollm-run --pretrain-input /mnt/data/train.txt --sft-input /mnt/data/distill_r1_110k_sft.jsonl --batch-size 2 --seq-len 1024 --accum 16 --pretrain-steps 61036
```

默认预训练 10000 步，SFT 2000 步；默认每 100 步保存检查点。首次运行前确定总步数。显存不足时可在新工作目录降低 batch-size/seq-len；原检查点严格保留原训练配置。脚本为单卡训练，多卡机器可用 `CUDA_VISIBLE_DEVICES=0 bash run.sh all` 选择一张卡，不支持 torchrun/DDP。

已有其他路径的预训练模型和 tokenizer 时：

```bash
bash run.sh sft --init-from runs/base/best.pt --tokenizer data/tokenizer.json
```

SSH 断开后继续运行（日志路径为示例）：

```bash
nohup bash run.sh all > cloud-console.log 2>&1 &
tail -f cloud-console.log
```

恢复时仍使用同一 work-dir。不要同时启动两个相同目录的任务；脚本用文件锁阻止并发写入。使用 Conda 时先激活环境；若 Python 命令不同，可设置 `PYTHON=/path/to/python bash run.sh all`。
