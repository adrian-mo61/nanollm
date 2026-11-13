"""Prepare short QA data, then initialize SFT from a pretrained checkpoint or resume."""
import argparse,json,sys
from pathlib import Path
from launch import run,prepare,train

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,default=Path('datasets/train.jsonl'))
 p.add_argument('--init-from',type=Path,default=Path('pretrain/best.pt'))
 p.add_argument('--tokenizer',type=Path,default=Path('sft/tokenizer.json'))
 p.add_argument('--work-dir',type=Path,default=Path('cloud_work_qa'))
 for name,value in [('steps',8000),('batch-size',8),('accum',4),('seq-len',256),('save-every',100),('eval-every',100)]: p.add_argument('--'+name,type=int,default=value)
 a=p.parse_args()
 if any(getattr(a,k)<=0 for k in ['steps','batch_size','accum','seq_len','save_every','eval_every']): p.error('Numeric options must be positive')
 import torch,hashlib
 if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required by this launcher; use train.py for CPU tests')
 work=a.work_dir.resolve(); work.mkdir(parents=True,exist_ok=True)
 import fcntl
 with (work/'.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  data=work/'data'; out=work/'sft'
  if not (out/'last.pt').exists() and not (out/'best.pt').exists():
   ck=torch.load(a.init_from,map_location='cpu',weights_only=False)
   if ck['meta'].get('kind')=='sft': raise ValueError('Use a pretrained checkpoint, not the old SFT checkpoint')
   assert hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()==ck['meta']['tokenizer_sha256'],'Tokenizer mismatch'
   del ck
  prepare(['prepare_sft.py','--input',a.input.resolve(),'--tokenizer',a.tokenizer.resolve(),'--output',data,'--seq-len',a.seq_len,'--deduplicate','--skip-long','--trim-padding'],data)
  meta=json.loads((data/'meta.json').read_text(encoding='utf-8'))
  assert meta.get('trim_padding') and meta['arguments']['skip_long'] and meta['arguments']['deduplicate'],'Use fresh QA data'
  a.seq_len=meta['seq_len']; a.sft_steps=a.steps
  train('sft',data,out,a,a.init_from.resolve())
if __name__=='__main__': main()
