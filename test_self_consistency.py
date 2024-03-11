"""Test self-consistency of StructFormer."""

import argparse
import collections

import matplotlib.pyplot as plt
from nltk.parse import DependencyGraph
import numpy
import torch

import data_ptb
import tree_utils
from hinton import plot


def mean(x):
  return sum(x) / len(x)


@torch.no_grad()
def test(parser_1, parser_2, corpus, device, prt=False, gap=0):
  """Compute UF1 and UAS scores.

  Args:
    parser: pretrained model
    corpus: labeled corpus
    device: cpu or gpu
    prt: bool, whether print examples
    gap: distance gap for building non-binary tree
  Returns:
    UF1: unlabeled F1 score for constituency parsing
  """
  parser_1.eval()
  parser_2.eval()

  prec_list = []
  reca_list = []
  f1_list = []
  
  dtree_list_1 = []
  dtree_list_2 = []

  word2idx = corpus.dictionary.word2idx
  dataset = zip(corpus.test_sens, corpus.test_trees, corpus.test_nltktrees)
  
  for sen, sen_tree, sen_nltktree in dataset:
    x = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in sen]
    data = torch.LongTensor([x]).to(device)
    pos = torch.LongTensor([list(range(len(sen)))]).to(device)

    
    # make the model 1 pass
    _, p_dict_1 = parser_1(data, pos)
    block_1 = p_dict_1['block']
    cibling_1 = p_dict_1['cibling']
    head_1 = p_dict_1['head']
    distance_1 = p_dict_1['distance']
    height_1 = p_dict_1['height']

    distance_1 = distance_1.clone().squeeze(0).cpu().numpy().tolist()
    height_1 = height_1.clone().squeeze(0).cpu().numpy().tolist()
    head_1 = head_1.clone().squeeze(0).cpu().numpy()
    max_height_1 = numpy.max(height_1)

    parse_tree_1 = tree_utils.build_tree(distance_1, sen, gap=gap)
    ##################################################################
    
    # make the model 2 pass
    _, p_dict_2 = parser_2(data, pos)
    block_2 = p_dict_2['block']
    cibling_2 = p_dict_2['cibling']
    head_2 = p_dict_2['head']
    distance_2 = p_dict_2['distance']
    height_2 = p_dict_2['height']

    distance_2 = distance_2.clone().squeeze(0).cpu().numpy().tolist()
    height_2 = height_2.clone().squeeze(0).cpu().numpy().tolist()
    head_2 = head_2.clone().squeeze(0).cpu().numpy()
    max_height_2 = numpy.max(height_2)

    parse_tree_2 = tree_utils.build_tree(distance_2, sen, gap=gap)
    ##################################################################

    model_out_1, _ = tree_utils.get_brackets(parse_tree_1)
    model_out_2, _ = tree_utils.get_brackets(parse_tree_2)
    
    overlap = model_out_1.intersection(model_out_2)

    prec = float(len(overlap)) / (len(model_out_1) + 1e-8)
    reca = float(len(overlap)) / (len(model_out_2) + 1e-8)
    
    # if not std_out:
    #   reca = 1.
    #   if not model_out:
    #     prec = 1.
    
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    prec_list.append(prec)
    reca_list.append(reca)
    f1_list.append(f1)

    
    
    # build dependency trees from predictions
    
    # model 1
    new_words = []
    true_words = sen_nltktree.pos()
    for d, c, w, ph in zip(distance_1, height_1, sen, head_1):
      next_word = true_words.pop(0)
      while next_word[1] not in data_ptb.WORD_TAGS:
        next_word = true_words.pop(0)
      new_words.append({
          'address': len(new_words) + 1,
          'word': next_word[0],
          'lemma': None,
          'ctag': None,
          'tag': next_word[1],
          'feats': None,
          'head': numpy.argmax(ph) + 1 if c < max_height_1 else 0,
          'deps': collections.defaultdict(list),
          'rel': None,
          'distance': d,
          'height': c
      })
    while true_words:
      next_word = true_words.pop(0)
      assert next_word[1] not in data_ptb.WORD_TAGS

    dtree = DependencyGraph()
    for w in new_words:
      dtree.add_node(w)

    dtree_list_1.append(dtree)


  # model 2
    new_words = []
    true_words = sen_nltktree.pos()
    for d, c, w, ph in zip(distance_2, height_2, sen, head_2):
      next_word = true_words.pop(0)
      while next_word[1] not in data_ptb.WORD_TAGS:
        next_word = true_words.pop(0)
      new_words.append({
          'address': len(new_words) + 1,
          'word': next_word[0],
          'lemma': None,
          'ctag': None,
          'tag': next_word[1],
          'feats': None,
          'head': numpy.argmax(ph) + 1 if c < max_height_2 else 0,
          'deps': collections.defaultdict(list),
          'rel': None,
          'distance': d,
          'height': c
      })
    while true_words:
      next_word = true_words.pop(0)
      assert next_word[1] not in data_ptb.WORD_TAGS

    dtree = DependencyGraph()
    for w in new_words:
      dtree.add_node(w)

    dtree_list_2.append(dtree)
  
  
  print('Constituency parsing performance:')
  print('Mean Prec: %.4f, Mean Reca: %.4f, Mean F1: %.4f' %
        (mean(prec_list), mean(reca_list), mean(f1_list)))
  
  print('-' * 89)

  print('Dependency parsing performance:')
  print('Conllu Style:')
  tree_utils.evald_self_cons(dtree_list_1, dtree_list_2, directed=True)
  tree_utils.evald_self_cons(dtree_list_1, dtree_list_2, directed=False)

  return mean(f1_list)


if __name__ == '__main__':
  marks = [' ', '-', '=']

  numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

  argpr = argparse.ArgumentParser(description='PyTorch PTB Language Model')

  # Model parameters.
  argpr.add_argument(
      '--data',
      type=str,
      default='data/penn/',
      help='location of the data corpus')
  argpr.add_argument(
      '--checkpoint_1',
      type=str,
      default='PTB.pt',
      help='model checkpoint to use')
  argpr.add_argument(
      '--checkpoint_2',
      type=str,
      default='PTB.pt',
      help='model checkpoint to use')
  argpr.add_argument('--seed', type=int, default=1111, help='random seed')
  argpr.add_argument('--gap', type=float, default=0, help='random seed')
  argpr.add_argument('--print', action='store_true', help='use CUDA')
  argpr.add_argument('--cuda', action='store_true', help='use CUDA')
  argpr.add_argument('--wsj10', action='store_true', help='use WSJ10')
  args = argpr.parse_args()

  # Set the random seed manually for reproducibility.
  torch.manual_seed(args.seed)

  print(f'Testing self-consistency of StructFormer using model {args.checkpoint_1} and model {args.checkpoint_2}')
  
  # Load model 1
  print('Loading model 1...')
  with open(args.checkpoint_1, 'rb') as f:
    model_1, _, _, _ = torch.load(f)
    torch.cuda.manual_seed(args.seed)
    model_1.cpu()
    if args.cuda:
      model_1.cuda()

  # Load model 2
  print('Loading model 2...')
  with open(args.checkpoint_2, 'rb') as f:
    model_2, _, _, _ = torch.load(f)
    torch.cuda.manual_seed(args.seed*1000)
    model_2.cpu()
    if args.cuda:
      model_2.cuda()


  # Load data
  print('Loading PTB dataset...')
  ptb_corpus = data_ptb.Corpus(args.data)

  print('Evaluating...')
  if args.cuda:
    eval_device = torch.device('cuda:0')
  else:
    eval_device = torch.device('cpu')

  print('=' * 89)
  test(model_1, model_2, ptb_corpus, eval_device, prt=args.print, gap=args.gap)
  print('=' * 89)
