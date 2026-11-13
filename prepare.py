import argparse, hashlib, json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tqdm import tqdm

def split(text, ratio):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), 'little') / 2**64 < ratio

def lines(path, ratio, train_only=False, limit=0):
    with open(path, 'rb') as f, tqdm(total=Path(path).stat().st_size, unit='B', unit_scale=True, desc='读取语料') as bar:
        for i, raw in enumerate(f):
            if limit and i >= limit: break
            bar.update(len(raw))
            s = raw.decode('utf-8').strip()
            if s and (not train_only or not split(s, ratio)): yield s

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', default='datasets/train.txt'); p.add_argument('--output', default='data')
    p.add_argument('--vocab-size', type=int, default=32000); p.add_argument('--val-ratio', type=float, default=.01)
    p.add_argument('--tokenizer-lines', type=int, default=1000000); p.add_argument('--limit-lines', type=int, default=0)
    a=p.parse_args(); assert 260 <= a.vocab_size <= 65536 and 0 < a.val_ratio < 1
    out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'meta.json').exists(): raise FileExistsError('输出目录已有数据，请使用新目录')
    tok=Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder=decoders.ByteLevel()
    tok.train_from_iterator(lines(a.input,a.val_ratio,True,min(a.tokenizer_lines,a.limit_lines) if a.limit_lines else a.tokenizer_lines), trainers.BpeTrainer(vocab_size=a.vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['<pad>','<unk>','<bos>','<eos>']))
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

