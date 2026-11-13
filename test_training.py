import subprocess, sys, tempfile
from pathlib import Path
import torch

def run(args): subprocess.run([sys.executable]+args,check=True)
root=Path(tempfile.mkdtemp(prefix='nanollm_test_'))
pre=str(root/'pre'); sft=str(root/'sft')
run(['prepare.py','--output',pre,'--limit-lines','300','--tokenizer-lines','300','--vocab-size','512','--val-ratio','.2'])
run(['prepare_sft.py','--tokenizer',str(Path(pre)/'tokenizer.json'),'--output',sft,'--limit-lines','100','--seq-len','64'])
for mode in ['pre','sft']:
    base=(['train.py','--data',pre,'--tiny','--seq-len','32'] if mode=='pre' else ['sft.py','--data',sft,'--seq-len','64'])+['--steps','3','--batch-size','1','--accum','1','--eval-every','1','--eval-batches','2','--warmup','1']
    init=[] if mode=='pre' else ['--init-from',str(root/'test_pre_full'/'best.pt')]
    run(base+['--out',str(root/f'test_{mode}_full')]+init)
    run(base+['--out',str(root/f'test_{mode}_resume'),'--stop-after','1']+init)
    run(base+['--out',str(root/f'test_{mode}_resume'),'--resume',str(root/f'test_{mode}_resume'/'last.pt')])
    a=torch.load(str(root/f'test_{mode}_full'/'last.pt'),weights_only=False); b=torch.load(str(root/f'test_{mode}_resume'/'last.pt'),weights_only=False)
    for k in a['model']: torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
    assert [r['train_loss'] for r in a['history']]==[r['train_loss'] for r in b['history']]
    print(mode,'resume exact equality OK')
from model import LM
from torch.nn import functional as F
m=LM(vocab=512,d=64,layers=2,heads=4,hidden=128).eval(); x=torch.randint(0,512,(2,32)); y=x.clone(); y[:,:20]=-100
expected=F.cross_entropy(m(x).reshape(-1,512),y.flatten())
torch.testing.assert_close(m(x,y),expected); print('masked loss OK')

from prepare_sft import parse_record
assert parse_record(dict(instruction='问题',input='背景',output='<think>思考</think>答案')) == ('问题\n\n背景','<think>思考</think>答案')
assert parse_record(dict(instruction='问题',input='',output='<think>思考</think>答案'),True) == ('问题','答案')
try: parse_record(dict(instruction='问题',output='<think>未闭合'),True)
except ValueError: pass
else: raise AssertionError('Malformed think must fail')
print('R1 parser OK; test artifacts:',root)
