import json

def open_file(path):
    lines = []
    with open(path, 'r',encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines



def main():
    # test, valid
    tp = 'valid'
    # copynet  seq2seqkbfinal
    model = 'seq2seqkbfinal'
    # s2s, cp, s2s_atte
    name = 's2s'
    src_path = f'projects/data/ori/{tp}_source.txt'
    tgt_path = f'projects/data/ori/{tp}_target.txt'
    pre_path = f'/data/yy/projects/rrg_others/rrg_2019/code-py3/output/{model}_greedy_review_{name}_{tp}.txt'
    src = open_file(src_path)
    tgt = open_file(tgt_path)
    pre_path = open_file(pre_path)
    res = []
    for idx in range(len(src)):
        res.append({
            'src': src[idx],
            'tgt': tgt[idx],
            'pred': pre_path[idx]
        })
    json.dump(res, open(f'eval/res_s2s_{tp}.json','w',encoding='utf-8'),ensure_ascii=False, indent=4)
    
    
main()