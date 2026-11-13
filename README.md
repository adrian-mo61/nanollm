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

### 预处理进度停在 4% 左右

旧版词表训练只读前 100 万行，但进度总量显示全文件字节数，因此可能停在约 558 MB / 12.7 GB。随后仍会进行 CPU BPE 计算；这时 GPU 空闲正常，旧 ETA 不代表词表训练剩余时间。新版按样本行数显示进度、打印阶段切换，并每 30 秒输出词表训练尚未返回的提示（不是计算完成百分比）。

在另一个终端运行 `top -c`、`free -h`，观察 prepare.py 的 CPU、内存及 swap；只凭静止进度不能判断卡死。若内存耗尽或大量换页，可停止当前进程后减少词表样本规模，例如 `bash run.sh all --work-dir cloud_work_small_vocab --tokenizer-lines 100000`。这只减少训练 BBPE 词表的样本，不减少后续全量语料分词；可能改变分词质量。不要替换已有模型使用的 tokenizer。

正在运行的进程不会自动应用 git pull 后的改动。若资源仍在正常工作可继续等待；需要新版提示时停止旧任务后再启动。预处理不支持中途续接，非空且缺少 meta.json 的目录需要保留并换用新的 work-dir。

## 使用 train.jsonl 从预训练模型重新做短问答 SFT

此流程支持 `question/answer` 格式，默认过滤空内容，去除相同问题与回答的完全重复项；按问题哈希分配 95%/5% 训练/验证。相同问题不同回答仍会保留，但不会跨集合。不会自动判断事实正确性或清除近似重复。

默认最多 256 个输入位置（记录包含额外一个右移标签 token），超过上限的样本完整跳过并记录，不截断、不拆开问题与回答。二进制仍为定长 uint16 ID + uint8 mask，训练每个 batch 按最后一个有效目标位置裁掉尾部 padding，仅监督回答和 EOS。不同样本之间不串接。

Linux 云 GPU 一键运行：

```bash
git pull --ff-only
# 本地文件布局：pretrain/best.pt、sft/tokenizer.json、datasets/train.jsonl
bash run.sh qa

# 若预训练结果仍在云服务器原目录中：
bash run.sh qa --init-from cloud_work/pretrain/best.pt --tokenizer cloud_work/pretrain_data/tokenizer.json
```

新输出为 cloud_work_qa/data 和 cloud_work_qa/sft，原预训练和旧 SFT 不覆盖。启动器拒绝将 SFT 检查点作为首次初始化，核对 tokenizer 哈希后仅加载预训练权重，重新创建优化器。默认 steps=8000、batch-size=8、accum=4、lr=2e-5、warmup=100、seq-len=256，每 100 步验证和保存。8000 步随机抽取约 256000 条记录，仍是有放回采样，不是严格一个 epoch；这是第一轮对照预算，不保证充分收敛。显存使用需在你的 GPU 实测。

再次运行相同命令自动恢复新目录的 last.pt（缺失时 best.pt），并恢复原训练日程及随机状态；不会重新从预训练权重开始。开始另一组实验应使用新的 --work-dir。改变 seq-len 需要重新预处理。

```bash
nohup bash run.sh qa > qa-console.log 2>&1 &
# 训练后推理；tokenizer 自动复制到训练输出目录
python generate.py --checkpoint cloud_work_qa/sft/best.pt --tokenizer cloud_work_qa/sft/tokenizer.json --prompt '中国的首都是哪里？只回答城市名称。' --max-new-tokens 128
```

分步运行或在 Windows 使用：

```bash
python prepare_sft.py --input datasets/train.jsonl --tokenizer sft/tokenizer.json --output qa_data --seq-len 256 --deduplicate --skip-long --trim-padding
python sft.py --data qa_data --out runs/qa --init-from pretrain/best.pt --seq-len 256 --batch-size 8 --accum 4 --steps 8000 --save-every 100
# 恢复，日程参数保持一致
python sft.py --data qa_data --out runs/qa --resume runs/qa/last.pt --seq-len 256 --batch-size 8 --accum 4 --steps 8000 --save-every 100
```

新数据 meta.json 记录去重、无效和超长数量，rejected.jsonl 记录无效/超长样本行号。动态裁剪模式下 metrics.csv 的 tokens_seen/tokens_per_sec 统计实际送入模型的位置（仍包含 batch 内 padding），额外记录 supervised_tokens_seen，表示累计有效回答及 EOS 标签数。旧数据的统计口径保持不变。

## 20M 小模型探索

新增 `--model-size 20m`：7 层、d=320、5 个注意力头（head_dim=64）、SwiGLU hidden=1024、共享 embedding/output。32K 词表下总参数 **19,993,280**，其中词嵌入10,240,000。结构仍为 Dense、RoPE、RMSNorm、因果 SDPA。默认不传 model-size 时仍是原100M；SFT/续训自动读取 checkpoint 结构，显式指定不匹配的 model-size 会报错。

复用已有32K BBPE和bin可以保持与100M实验的分词条件一致，无需重新预处理；但约一半参数用于词表，这是当前对照设计的权衡。20M须从随机权重预训练，不能加载100M权重。模型更小是为了降低探索成本，不意味着效果会更好。

Linux 云 GPU 第一轮：固定32768 token/step，预训练约4亿token（12208步），作为实验预算而非效果保证。请确保现有 pretrain_data 目录含 train.bin、val.bin、meta.json、tokenizer.json。

```bash
git pull --ff-only
bash run.sh pretrain --model-size 20m --pretrain-data cloud_work/pretrain_data --work-dir cloud_work_20m --pretrain-steps 12208 --batch-size 8 --seq-len 512 --accum 8
```

若你的二进制目录实际是 pretrain_data/，替换命令中的 cloud_work/pretrain_data。若尚无bin，去掉 --pretrain-data，会在新的工作目录中训练词表并处理原始语料。

预训练完成后，在同一20M模型上做短问答SFT：

```bash
bash run.sh qa --init-from cloud_work_20m/pretrain/best.pt --tokenizer cloud_work_20m/pretrain/tokenizer.json --work-dir cloud_work_20m_qa
python generate.py --checkpoint cloud_work_20m_qa/sft/best.pt --tokenizer cloud_work_20m_qa/sft/tokenizer.json --prompt '中国的首都是哪里？只回答城市名称。' --max-new-tokens 128
```

相同命令再次运行会恢复对应目录的检查点。保持各实验工作目录独立；不要用旧100M的训练目录运行20M。当前严格resume仍不支持改变steps或batch；预先确定预算，或者在新目录进行独立实验。

直接训练接口（不走Linux启动器）：

```bash
python train.py --model-size 20m --data pretrain_data --out runs/20m --steps 12208 --batch-size 8 --seq-len 512 --accum 8 --save-every 100
```

建议先完成一组20M基线，再分别改变预训练预算或SFT数据，每次只改一个因素。记录参数量、token预算、数据版本、采样策略、验证loss及固定问答的正确性/格式/EOS/重复表现。这里512上下文与此前100M的1024不同，初轮结果不是严格的参数量消融；做严格大小对比时，应统一上下文、词表、token预算和SFT设置。已有测试表明短回答SFT改善输出结束行为，不能据此推断20M能获得更强推理能力。
