"""Single-GPU pipeline with automatic checkpoint resume."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(args):
    print('Launching:', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, '-u', *map(str,args)], cwd=ROOT, check=True)

def prepare(args, directory):
    if (directory/'meta.json').is_file():
        print('Reusing prepared data:', directory)
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f'Incomplete preprocessing in {directory}; select a new --work-dir or move that directory aside before retrying.')
    run(args)

def train(stage, data, out, args, init=None):
    import torch
    checkpoint=out/'last.pt'
    if not checkpoint.exists() and (out/'best.pt').exists(): checkpoint=out/'best.pt'
    command=['sft.py' if stage=='sft' else 'train.py','--data',data,'--out',out]
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        # Restore the original schedule rather than launcher defaults.
        signature=ck['signature']
        for key,value in signature.items(): command.extend(['--'+key.replace('_','-'),value])
        command.extend(['--resume',checkpoint])
        print(f'Resuming {stage} at step {ck["step"]}/{signature["steps"]}')
        del ck
    else:
        command.extend(['--steps',args.sft_steps if stage=='sft' else args.pretrain_steps,
                        '--lr',2e-5 if stage=='sft' else 3e-4,
                        '--warmup',100 if stage=='sft' else 500,
                        '--batch-size',args.batch_size,'--accum',args.accum,'--seq-len',args.seq_len])
        if stage=='sft':
            if not init or not init.is_file(): raise FileNotFoundError(f'Pretrained checkpoint missing: {init}')
            command.extend(['--init-from',init])
    command.extend(['--save-every',args.save_every,'--eval-every',args.eval_every])
    run(command)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['all','pretrain','sft'],nargs='?',default='all')
    p.add_argument('--work-dir',type=Path,default=Path('cloud_work'))
    p.add_argument('--pretrain-input',type=Path,default=Path('datasets/train.txt'))
    p.add_argument('--sft-input',type=Path,default=Path('datasets/distill_r1_110k_sft.jsonl'))
    p.add_argument('--tokenizer-lines',type=int,default=1000000)
    p.add_argument('--init-from',type=Path); p.add_argument('--tokenizer',type=Path)
    for name,value in [('pretrain-steps',10000),('sft-steps',2000),('batch-size',2),('accum',16),('seq-len',1024),('save-every',100),('eval-every',100)]:
        p.add_argument('--'+name,type=int,default=value)
    a=p.parse_args()
    if a.tokenizer_lines<0: p.error('tokenizer-lines must be >= 0')
    if sys.version_info<(3,11): p.error('Python >= 3.11 required')
    for key in ['pretrain_steps','sft_steps','batch_size','accum','seq_len','save_every','eval_every']:
        if getattr(a,key)<=0: p.error(f'{key} must be positive')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; choose a cloud CUDA PyTorch image and check nvidia-smi.')
    # Exercise the actual CUDA runtime before expensive preprocessing.
    (torch.ones(8,device='cuda')+1).sum().item()
    print('GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__,flush=True)
    work=a.work_dir.resolve(); work.mkdir(parents=True,exist_ok=True)
    # Linux process lock prevents two launchers writing the same run.
    import fcntl
    with (work/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pre=work/'pretrain_data'; sft=work/'sft_data'; preout=work/'pretrain'; sftout=work/'sft'
        if a.stage in ['all','pretrain']:
            prepare(['prepare.py','--input',a.pretrain_input.resolve(),'--output',pre,'--tokenizer-lines',a.tokenizer_lines],pre)
            train('pretrain',pre,preout,a)
        if a.stage in ['all','sft']:
            tokenizer=a.tokenizer.resolve() if a.tokenizer else pre/'tokenizer.json'
            if not tokenizer.is_file(): raise FileNotFoundError('Pass --tokenizer for an existing pretrained model')
            prepare(['prepare_sft.py','--input',a.sft_input.resolve(),'--tokenizer',tokenizer,'--output',sft,'--seq-len',a.seq_len],sft)
            train('sft',sft,sftout,a,a.init_from.resolve() if a.init_from else preout/'best.pt')
    print('Completed. Artifacts:',work)

if __name__=='__main__': main()
