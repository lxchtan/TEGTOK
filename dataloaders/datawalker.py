import os
import json

class DatasetWalker(object):
  """
    history: [str, ]
    label: {'response': str, 'emotion': str (single word)}
  """
  def __init__(self, dataset, dataroot, labels, debug=False):
    self.labels = labels

    path = os.path.join(os.path.abspath(dataroot))

    if dataset not in ['train', 'valid', 'test']:
      raise ValueError('Wrong dataset name: %s' % (dataset))

    file = os.path.join(path, f'{dataset}.json')
    with open(file, 'r') as f:
      examples = json.load(f)

    # Reduce the time of loading dataset when debugging
    if debug:
      examples = examples[:2000]

    self.data = []
    if dataset in ['train', 'valid', 'test']:
      for example in examples:
        new_example = {}
        
        # Currently, only the top-1 knowledge entry is used
        new_example["dialog_id"] = example["dialog_id"]
        new_example["knowledge"] = {
          'wiki': [" ".join([t, k]) for t, k in zip(example["wiki_knowledge_title"], example["wiki_knowledge"])],
          'reddit': [" ".join([t, k]) for t, k in zip(example["reddit_knowledge_title"], example["reddit_knowledge"])],
        }

        new_example["history"] = example["post"]
        new_example["response"] = example["response"]

        self.data.append(new_example)
