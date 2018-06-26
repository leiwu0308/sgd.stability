import os
import pickle

import torch
from src.utils import load_model, load_data, parse_args
from src.folderwalker import scan
from src.net_utils import num_parameters



def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid
    trDL,teDL = load_data(args,stop=True,one_hot=True)
    net = load_model(args.dataset,args.arch,width=args.width,depth=args.depth)
    ct = torch.nn.MSELoss()
    print('# of parameters',num_parameters(net))

    res = scan(net,ct,trDL,teDL,args.model,verbose=True,
                    niters=50,nonuniformity=args.nonuniformity)
    with open(args.save_res,'wb') as f:
        pickle.dump(res,f)

if __name__ == '__main__':
    main()
