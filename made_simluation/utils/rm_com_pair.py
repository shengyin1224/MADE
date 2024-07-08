import torch 

def rm_com_pair(com_src: torch.Tensor, com_tgt: torch.Tensor, 
                rm_src: torch.Tensor, rm_tgt: torch.Tensor):
    lpt = set()
    for src, tgt in zip(rm_src, rm_tgt):
        lpt.add((src[0].item(), src[1].item(), tgt[1].item()))
    
    new_com_src = []
    new_com_tgt = []
    for src, tgt in zip(com_src, com_tgt):
        if (src[0].item(), src[1].item(), tgt[1].item()) not in lpt:
            new_com_src.append(src)
            new_com_tgt.append(tgt)
    
    new_com_src = torch.stack(new_com_src, dim=0)
    new_com_tgt = torch.stack(new_com_tgt, dim=0)

    return new_com_src, new_com_tgt

