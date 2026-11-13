import shutil
import argparse, contextlib, csv, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange
from model import LM

def main():
    p=argparse.ArgumentParser()
    for name,default in [('data','data'),('out','runs/base'),('resume','')]: p.add_argument('--'+name,default=default)
    for name,default in [('steps',10000),('batch-size',2),('seq-len',1024),('accum',16),('eval-every',100),('eval-batches',20),('save-every',500),('warmup',500),('seed',42)]: p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--init-from',default=''); p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--tiny',action='store_true')
    a=p.parse_args()
    assert all(getattr(a,k)>0 for k in ['steps','batch_size','seq_len','accum','eval_every','eval_batches','save_every'])
    torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.csv').exists() and not a.resume: raise FileExistsError('请更换输出目录或使用 --resume')
    data=Path(a.data); meta=json.loads((data/'meta.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((data/'tokenizer.json').read_bytes()).hexdigest()==meta['tokenizer_sha256']
    is_sft=meta.get('kind')=='sft'
    if is_sft:
        assert a.seq_len==meta['seq_len'], 'seq-len 必须与 SFT 数据一致'
        for name,digest in meta['files_sha256'].items():
            with open(data/name,'rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest, 'SFT 文件损坏'
    arrays={k:np.memmap(data/(k+'.bin'),dtype=meta['dtype'],mode='r') for k in ['train','val']}
    for k,v in arrays.items():
        assert len(v)==(meta['records'][k]*(a.seq_len+1) if is_sft else meta['tokens'][k]) and len(v)>a.seq_len, f'{k} 数据不足或损坏'
    masks={k:np.memmap(data/(k+'.mask.bin'),dtype='u1',mode='r').reshape(-1,a.seq_len+1) for k in arrays} if is_sft else {}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    def amp(): return torch.autocast('cuda',dtype=dtype) if device=='cuda' else contextlib.nullcontext()
    cfg=dict(vocab=meta['vocab_size'],d=64,layers=2,heads=4,hidden=128) if a.tiny else dict(vocab=meta['vocab_size'],d=768,layers=12,heads=12,hidden=1792)
    initial=None
    if is_sft:
        assert a.resume or a.init_from, 'SFT 必须提供 --init-from 预训练模型或 --resume SFT检查点'
    if a.init_from:
        assert not a.resume, '--init-from 与 --resume 不能同时使用'
        initial=torch.load(a.init_from,map_location='cpu',weights_only=False)
        assert initial['meta']['tokenizer_sha256']==meta['tokenizer_sha256'], '必须复用预训练 tokenizer'
        cfg=initial['config']
    if a.resume:
        cfg=torch.load(a.resume,map_location='cpu',weights_only=False)['config']
    model=LM(**cfg).to(device); n=sum(x.numel() for x in model.parameters())
    if initial is not None: model.load_state_dict(initial['model'])
    opt=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.1},{'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}],lr=a.lr,betas=(.9,.95))
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and dtype==torch.float16)
    start=0; best=float('inf'); history=[]
    signature={k:getattr(a,k) for k in ['batch_size','seq_len','accum','steps','warmup','lr','seed']}
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False)
        assert ck['config']==cfg and ck['meta']==meta and ck['signature']==signature, '续训配置或数据不一致'
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start=ck['step']; best=ck['best']; history=ck['history']; rng.bit_generator.state=ck['numpy_rng']; torch.set_rng_state(ck['torch_rng'].cpu())
        if device=='cuda' and ck['cuda_rng']: torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    (out/'config.json').write_text(json.dumps(dict(arguments=vars(a),model=cfg,parameters=n,device=device,torch=torch.__version__,python=platform.python_version(),data=meta),indent=2),encoding='utf-8')
    if (data/'tokenizer.json').resolve() != (out/'tokenizer.json').resolve():
        shutil.copyfile(data/'tokenizer.json',out/'tokenizer.json')
    def batch(key,g):
        if is_sft:
            idx=g.integers(0,meta['records'][key],size=a.batch_size)
            raw=arrays[key].reshape(-1,a.seq_len+1)[idx]
            mask=masks[key][idx]
            end=int(np.nonzero(mask)[1].max())+1 if meta.get('trim_padding') else a.seq_len+1
            z=torch.from_numpy(raw[:,:end].astype(np.int64)).to(device)
            y=z[:,1:].clone(); weights=torch.from_numpy(mask[:,1:end].copy()).to(device)
            y[weights==0]=-100
            return z[:,:-1],y
        v=arrays[key]; starts=g.integers(0,len(v)-a.seq_len,size=a.batch_size)
        z=torch.from_numpy(np.stack([v[i:i+a.seq_len+1].astype(np.int64) for i in starts])).to(device)
        return z[:,:-1],z[:,1:]
    def save(name,step):
        tmp=out/(name+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),config=cfg,meta=meta,signature=signature,step=step,best=best,history=history,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []),tmp)
        tmp.replace(out/name)
    def report():
        if not history: return
        with open(out/'metrics.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        fig,axs=plt.subplots(2,2,figsize=(11,7))
        for key,ax in zip(['train_loss','val_loss','tokens_per_sec','grad_norm'],axs.flat):
            rows=[r for r in history if r[key] is not None]; ax.plot([r['step'] for r in rows],[r[key] for r in rows]); ax.set(title=key,xlabel='step'); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out/'loss_and_metrics.png',dpi=160); plt.close(fig)
    print(f'Parameters: {n:,}; device: {device}')
    tokens=a.batch_size*a.seq_len*a.accum
    bar=trange(start+1,a.steps+1,desc='SFT' if is_sft else '预训练')
    for step in bar:
        if device=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter(); model.train(); opt.zero_grad(set_to_none=True)
        lr=a.lr*min(1.,step/max(1,a.warmup))*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-a.warmup)/max(1,a.steps-a.warmup))))
        for group in opt.param_groups: group['lr']=lr
        loss_sum=0.; actual_positions=0; supervised_positions=0
        for _ in range(a.accum):
            bx,by=batch('train',rng)
            actual_positions+=bx.numel(); supervised_positions+=(by!=-100).sum().item()
            with amp(): loss=model(bx,by)
            loss_sum+=loss.detach().float().item(); scaler.scale(loss/a.accum).backward()
        scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.).item()
        if not math.isfinite(loss_sum) or not math.isfinite(norm): raise FloatingPointError('非有限 loss/梯度，训练已停止')
        scaler.step(opt); scaler.update()
        if device=='cuda': torch.cuda.synchronize()
        elapsed=time.perf_counter()-tick; tokens=a.batch_size*a.seq_len*a.accum; val=None
        if step%a.eval_every==0 or step==a.steps:
            model.eval(); vrng=np.random.default_rng(a.seed+1); losses=[]
            with torch.no_grad(),amp():
                for _ in range(a.eval_batches):
                    vx,vy=batch('val',vrng); weight=(vy!=-100).sum().item(); losses.append((model(vx,vy).item(),weight))
            val=sum(l*w for l,w in losses)/sum(w for _,w in losses)
        history.append(dict(step=step,tokens_seen=step*tokens,train_loss=loss_sum/a.accum,val_loss=val,val_ppl=math.exp(min(val,80)) if val is not None else None,lr=lr,grad_norm=norm,tokens_per_sec=tokens/elapsed,step_seconds=elapsed,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9 if device=='cuda' else 0))
        if meta.get('trim_padding'):
            previous=history[-2] if len(history)>1 else {}
            history[-1].update(tokens_seen=previous.get('tokens_seen',0)+actual_positions,tokens_per_sec=actual_positions/elapsed,supervised_tokens_seen=previous.get('supervised_tokens_seen',0)+supervised_positions)
        bar.set_postfix(loss=f'{loss_sum/a.accum:.4f}',val=val, tok_s=int(history[-1]['tokens_per_sec']))
        if val is not None:
            report()
            if val<best: best=val; save('best.pt',step)
        if step%a.save_every==0 or step==a.steps: save('last.pt',step); report()
        if a.stop_after and step>=a.stop_after:
            save('last.pt',step); report(); break
    (out/'summary.json').write_text(json.dumps(dict(parameters=n,best_val_loss=best,best_val_ppl=math.exp(min(best,80)),steps=history[-1]['step'] if history else start,tokens_seen=history[-1]['tokens_seen'] if history else 0),indent=2),encoding='utf-8')
if __name__=='__main__': main()
