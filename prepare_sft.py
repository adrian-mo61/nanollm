"""SFT bin format: fixed-length records of seq_len+1 uint16 IDs and uint8 loss masks."""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
from prepare import split

def prompt_text(text):
    return 'User:\n'+text+'\n\nAssistant:\n'

def parse_record(obj, answer_only=False):
    instruction = obj.get('question') if 'question' in obj else obj.get('instruction')
    context = obj.get('input', '')
    answer = obj.get('answer') if 'question' in obj else obj.get('output')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction 必须是非空字符串')
    if not isinstance(context, str) or not isinstance(answer, str) or not answer.strip():
        raise ValueError('input 必须为字符串，output 必须为非空字符串')
    question = instruction.strip() + ('\n\n' + context.strip() if context.strip() else '')
    if answer_only and answer.lstrip().startswith('<think>'):
        if '</think>' not in answer:
            raise ValueError('缺少 </think>，无法提取最终回答')
        answer = answer.split('</think>', 1)[1].lstrip()
        if not answer: raise ValueError('最终回答为空')
    return question, answer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='datasets/distill_r1_110k_sft.jsonl')
    p.add_argument('--tokenizer',default='data/tokenizer.json'); p.add_argument('--output',default='sft_data')
    p.add_argument('--seq-len',type=int,default=1024); p.add_argument('--val-ratio',type=float,default=.05)
    p.add_argument('--deduplicate',action='store_true')
    p.add_argument('--skip-long',action='store_true')
    p.add_argument('--trim-padding',action='store_true')
    p.add_argument('--answer-only',action='store_true'); p.add_argument('--limit-lines',type=int,default=0)
    a=p.parse_args(); assert a.seq_len>=2 and 0<a.val_ratio<1
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('请使用空输出目录')
    tok=Tokenizer.from_file(a.tokenizer); assert tok.get_vocab_size()<=65536
    eos=tok.token_to_id('<eos>'); pad=tok.token_to_id('<pad>'); assert eos is not None and pad is not None
    shutil.copyfile(a.tokenizer,out/'tokenizer.json')
    rejected=0; duplicates=0; too_long=0; seen=set()
    counts={k:0 for k in ['train','val']}; docs=counts.copy(); supervised=counts.copy()
    from contextlib import ExitStack
    with ExitStack() as stack:
        files={k:(stack.enter_context(open(out/(k+'.bin'),'wb')),stack.enter_context(open(out/(k+'.mask.bin'),'wb'))) for k in counts}
        reject_log=stack.enter_context(open(out/'rejected.jsonl','w',encoding='utf-8'))
        f=stack.enter_context(open(a.input,'rb'))
        bar=stack.enter_context(tqdm(total=Path(a.input).stat().st_size,unit='B',unit_scale=True,desc='SFT tokenize'))
        for i,raw in enumerate(f):
            if a.limit_lines and i>=a.limit_lines: break
            bar.update(len(raw))
            try:
                obj=json.loads(raw)
                if not isinstance(obj,dict): raise ValueError('记录必须为 JSON 对象')
                question,answer=parse_record(obj,a.answer_only)
            except (ValueError, UnicodeError) as e:
                rejected+=1
                reject_log.write(json.dumps(dict(line=i+1,reason=str(e)),ensure_ascii=False)+'\n')
                continue
            pair=hashlib.sha256(json.dumps([question,answer],ensure_ascii=False).encode()).digest()
            if a.deduplicate and pair in seen:
                duplicates+=1
                continue
            if a.deduplicate: seen.add(pair)
            key='val' if split(question.strip(),a.val_ratio) else 'train'
            prefix=tok.encode(prompt_text(question),add_special_tokens=False).ids
            target=tok.encode(answer,add_special_tokens=False).ids+[eos]
            ids=prefix+target; mask=[0]*len(prefix)+[1]*len(target)
            if a.skip_long and len(ids)>a.seq_len+1:
                too_long+=1
                reject_log.write(json.dumps(dict(line=i+1,reason='too_long',tokens=len(ids)))+'\n')
                continue
            docs[key]+=1
            # Overlap one token so every answer token is predicted once, including chunk boundaries.
            for start in range(0,len(ids)-1,a.seq_len):
                chunk=ids[start:start+a.seq_len+1]; weights=mask[start:start+a.seq_len+1]
                if not any(weights[1:]): continue
                supervised[key]+=sum(weights[1:]); padding=a.seq_len+1-len(chunk)
                np.asarray(chunk+[pad]*padding,dtype='<u2').tofile(files[key][0])
                np.asarray(weights+[0]*padding,dtype='u1').tofile(files[key][1]); counts[key]+=1
    if not all(counts.values()): raise ValueError('训练或验证集为空，请增加样本/调整 val-ratio，并使用新目录重试')
    meta=dict(trim_padding=a.trim_padding,duplicate_records=duplicates,too_long_records=too_long,rejected_records=rejected,kind='sft',dtype='<u2',seq_len=a.seq_len,records=counts,documents=docs,supervised_tokens=supervised,vocab_size=tok.get_vocab_size(),tokenizer_sha256=hashlib.sha256((out/'tokenizer.json').read_bytes()).hexdigest(),answer_only=a.answer_only,format='qa_or_alpaca',arguments=vars(a))
    meta['files_sha256']={}
    for path in out.glob('*.bin'):
        with path.open('rb') as f: meta['files_sha256'][path.name]=hashlib.file_digest(f,'sha256').hexdigest()
    (out/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(meta)
if __name__=='__main__': main()
