import sys
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import LM
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='runs/base/best.pt'); p.add_argument('--tokenizer',default='data/tokenizer.json'); p.add_argument('--prompt',default='人工智能的发展'); p.add_argument('--max-new-tokens',type=int,default=100); p.add_argument('--temperature',type=float,default=.8); a=p.parse_args()
assert a.temperature>0
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
import hashlib
assert hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()==ck['meta']['tokenizer_sha256']
device='cuda' if torch.cuda.is_available() else 'cpu'; m=LM(**ck['config']).to(device); m.load_state_dict(ck['model']); m.eval(); tok=Tokenizer.from_file(a.tokenizer)
if ck['meta'].get('kind')=='sft':
    from prepare_sft import prompt_text
    a.prompt=prompt_text(a.prompt)
ids=tok.encode(a.prompt,add_special_tokens=False).ids or [tok.token_to_id('<bos>')]
x=torch.tensor([ids],device=device)
with torch.no_grad():
    for _ in range(a.max_new_tokens):
        logits=m(x[:,-ck['signature']['seq_len']:])[:,-1]/a.temperature
        nxt=torch.multinomial(logits.softmax(-1),1); x=torch.cat((x,nxt),1)
        if nxt.item()==tok.token_to_id('<eos>'): break
sys.stdout.reconfigure(encoding='utf-8')
print(tok.decode(x[0].tolist()))

