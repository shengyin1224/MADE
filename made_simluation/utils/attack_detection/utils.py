import torch

def label_attacker(com_src: torch.Tensor, com_tgt: torch.Tensor, 
                    attack_src: torch.Tensor, attack_tgt: torch.Tensor):
    if attack_src is None or attack_tgt is None:
        return torch.zeros(len(com_src), dtype=torch.long).to(com_src.device)
    lpt = {}    
    for src, tgt in zip(attack_src, attack_tgt):
        lpt[(src[0].item(), src[1].item(), tgt[1].item())] = 1

    label = []
    for src, tgt in zip(com_src, com_tgt):
        if (src[0].item(), src[1].item(), tgt[1].item()) in lpt:
            label.append(1)
        else:
            label.append(0)
    label = torch.Tensor(label).long().to(com_src.device)
    return label

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