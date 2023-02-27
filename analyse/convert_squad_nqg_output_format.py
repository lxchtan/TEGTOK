"""
[
  {
    "dialog_id": "test-5729",
    "ground_truth": "how does portugal compete with other areas for tourists ?",
    "generated": "what has been necessary for hyderabad to stay ahead of its competitors?"
  },
"""
import json

def convert(filepath):
  with open(filepath) as f:
    fdata = json.load(f)
  fdata.sort(key=lambda x: int(x['dialog_id'].split('-')[-1]))
  lines = [line['generated'].replace('?', ' ?') + '\n' for line in fdata]
  with open(filepath.replace('.json', '_format.txt'), 'w') as fw:
    fw.writelines(lines)

if __name__ == '__main__':
  for filename in [
    'results/20230227_TegTok_nqg_13_single_example.json',
    'results/20230227_TegTok_nqg_43_single_example.json',
    'results/20230227_TegTok_nqg_91_single_example.json',
    'results/20230227_TegTok_nqg_7677_single_example.json',
  ]:
    convert(filename)