"""Shared training engine; SFT defaults. Resume uses the same full schedule."""
import sys
from train import main
if __name__=='__main__':
    defaults={'--data':'sft_data','--out':'runs/sft','--lr':'2e-5','--warmup':'100','--steps':'2000'}
    for key,value in defaults.items():
        if not any(x==key or x.startswith(key+'=') for x in sys.argv): sys.argv.extend([key,value])
    main()
