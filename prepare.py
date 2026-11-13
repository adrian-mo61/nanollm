import argparse, hashlib, json, threading, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tqdm import tqdm

def split(text, ratio):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), 'little') / 2**64 < ratio

@contextmanager
def tokenizer_status(interval=30):
    done = threading.Event()
    started = time.monotonic()
    def report():
        while not done.wait(interval):
            print(f'[BBPE] CPU 词表训练尚未返回，已耗时 {time.monotonic()-started:.0f}s；此提示不是计算进度，资源使用请查看 top/free。', flush=True)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try: yield
    finally:
        done.set()
        thread.join()


def lines(path, ratio, train_only=False, limit=0):
    sampled = bool(limit)
    desc = '词表样本读取' if train_only else '语料分词读取'
    with open(path, 'rb') as f, tqdm(total=limit if sampled else Path(path).stat().st_size,
            unit='行' if sampled else 'B', unit_scale=True, desc=desc) as bar:
        for i, raw in enumerate(f):
            if limit and i >= limit: break
            bar.update(1 if sampled else len(raw))
            s = raw.decode('utf-8').strip()
            if s and (not train_only or not split(s, ratio)): yield s
    if train_only:
        print('词表样本读取完成；继续在 CPU 上统计及合并 BPE，尚未开始全量分词或 GPU 训练。', flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', default='datasets/train.txt'); p.add_argument('--output', default='data')
    p.add_argument('--vocab-size', type=int, default=32000); p.add_argument('--val-ratio', type=float, default=.01)
    p.add_argument('--tokenizer-lines', type=int, default=1000000); p.add_argument('--limit-lines', type=int, default=0)
    a=p.parse_args(); assert a.tokenizer_lines>=0 and a.limit_lines>=0
    assert 260 <= a.vocab_size <= 65536 and 0 < a.val_ratio < 1
    out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'meta.json').exists(): raise FileExistsError('输出目录已有数据，请使用新目录')
    tok=Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder=decoders.ByteLevel()
    limits=[x for x in [a.tokenizer_lines,a.limit_lines] if x>0]
    sample_limit=min(limits) if limits else 0
    print(f'[1/2] 训练 BBPE 词表（CPU），样本上限：{sample_limit or chr(8734)} 行。',flush=True)
    with tokenizer_status():
        tok.train_from_iterator(lines(a.input,a.val_ratio,True,sample_limit), trainers.BpeTrainer(vocab_size=a.vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['<pad>','<unk>','<bos>','<eos>']))
    print('[2/2] 词表已完成，开始将语料写入 train.bin / val.bin。',flush=True)
    tok.save(str(out/'tokenizer.json')); counts={'train':0,'val':0}; docs={'train':0,'val':0}
    with open(out/'train.bin','wb') as tr, open(out/'val.bin','wb') as va:
        batch=[]
        def flush():
            for s, enc in zip(batch,tok.encode_batch(batch,add_special_tokens=False)):
                key='val' if split(s,a.val_ratio) else 'train'
                ids=enc.ids+[tok.token_to_id('<eos>')]
                np.asarray(ids,dtype='<u2').tofile(va if key=='val' else tr)
                counts[key]+=len(ids); docs[key]+=1
            batch.clear()
        for s in lines(a.input,a.val_ratio,limit=a.limit_lines):
            batch.append(s)
            if len(batch)>=256: flush()
        if batch: flush()
    meta=dict(dtype='<u2', tokens=counts,documents=docs,vocab_size=tok.get_vocab_size(), tokenizer_sha256=hashlib.sha256((out/'tokenizer.json').read_bytes()).hexdigest(), source=str(Path(a.input).resolve()),arguments=vars(a))
    (out/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(meta)
if __name__=='__main__': main()
