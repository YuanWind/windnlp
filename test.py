from pyrouge import Rouge155
from scripts.utils import load_json
def split(path):
    data = load_json(path)
    for idx, inst in enumerate(data):
        tgt = inst['tgt']
        pred = inst['pred']
        with open(f'eval/tgt/clothing.{idx:03d}.txt','w',encoding='utf-8') as f:
            f.write(tgt)
        with open(f'eval/pred/clothing.A.{idx:03d}.txt','w',encoding='utf-8') as f:
            f.write(pred)

def main():
    r = Rouge155()
    r.system_dir = 'eval/tgt'
    r.model_dir = 'eval/pred'
    r.system_filename_pattern = 'clothing.(\d+).txt'
    r.model_filename_pattern = 'clothing.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)
    print(output_dict)
    
split('projects/outs/bart_zh/temp_dir/test_pred.json')
main()