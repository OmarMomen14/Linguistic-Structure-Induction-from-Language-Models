import os
import pickle
import re

import nltk
from nltk.corpus import ptb
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoTokenizer

WORD_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
    'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
    'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
    'WP$', 'WRB'
]
CURRENCY_TAGS_WORDS = ['#', '$', 'C$', 'A$']
ELLIPSIS = [
    '*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*'
]
PUNCTUATION_TAGS = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
PUNCTUATION_WORDS = [
    '.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!',
    '...', '-LCB-', '-RCB-'
]


class SubWord_Corpus(object):

  def __init__(self, tokenizer_name):
    """Initialization.

    Args:
      tokenizer_name: path to tokenizer
    Raises:
      Exception: missing dictionary
    """
    
    try:
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except:
      raise ValueError('Tokenizer does not exist')
    
    try:
      self.detok = TreebankWordDetokenizer()
    except:
      raise ValueError('TreeDetokenizer Error')

    all_file_ids = ptb.fileids()
    train_file_ids = []
    valid_file_ids = []
    test_file_ids = []
    rest_file_ids = []
    for file_id in all_file_ids:
      if 'WSJ/00/WSJ_0200.MRG' <= file_id <= 'WSJ/21/WSJ_2199.MRG':
        train_file_ids.append(file_id)
      if 'WSJ/22/WSJ_2200.MRG' <= file_id <= 'WSJ/22/WSJ_2299.MRG':
        valid_file_ids.append(file_id)
      if 'WSJ/23/WSJ_2300.MRG' <= file_id <= 'WSJ/23/WSJ_2399.MRG':
        test_file_ids.append(file_id)
      elif ('WSJ/00/WSJ_0000.MRG' <= file_id <= 'WSJ/01/WSJ_0199.MRG') or \
          ('WSJ/24/WSJ_2400.MRG' <= file_id <= 'WSJ/24/WSJ_2499.MRG'):
        rest_file_ids.append(file_id)

    self.train, self.train_sens, self.train_trees, self.train_nltktrees \
        = self.tokenize(train_file_ids)
    self.valid, self.valid_sens, self.valid_trees, self.valid_nltktress \
        = self.tokenize(valid_file_ids)
    self.test, self.test_sens, self.test_trees, self.test_nltktrees \
        = self.tokenize(test_file_ids)
    self.rest, self.rest_sens, self.rest_trees, self.rest_nltktrees \
        = self.tokenize(rest_file_ids)

  def tokenize(self, file_ids):
    """Tokenizes a mrg file."""

    NEG_POSS = ['n\'t', '\'s', '\'m', '\'re', '\'ve', '\'d', '\'ll', '%']

    def filter_words(tree):
        words = []
        for w, tag in tree.pos():
            if tag in WORD_TAGS:
                w = w.lower()
                words.append(w)
        return words

    def tree2list(tree):
        if isinstance(tree, nltk.Tree):
            if (tree.label() in WORD_TAGS):
                w = tree.leaves()[0]
                return w
            else:
                root = []
                for child in tree:
                    c = tree2list(child)
                    if c:
                        root.append(c)
                if len(root) > 1:
                    return root
                elif len(root) == 1:
                    return root[0]
        return []

    def merge_strings_in_tree(tree, neg_poss):
        def merge_with_last_string(node, string_to_merge):
            if isinstance(node[-1], str):
                node[-1] += string_to_merge
            elif isinstance(node[-1], list):
                merge_with_last_string(node[-1], string_to_merge)

        def merge_recursive(node, prev_node=None):
            if isinstance(node, list):
                new_node = []
                for item in node:
                    merged_item = merge_recursive(item, new_node if new_node else prev_node)
                    if merged_item is not None:
                        new_node.append(merged_item)
                return new_node
            else:
                if node.lower() in neg_poss and prev_node is not None:
                    merge_with_last_string(prev_node, node)
                    return None
                else:
                    return [node]

        def remove_wrapping_lists(node):
            if isinstance(node, list):
                if len(node) == 1 and isinstance(node[0], str):
                    return node[0]
                else:
                    return [remove_wrapping_lists(item) for item in node]
            return node

        merged_tree = merge_recursive(tree)
        return remove_wrapping_lists(merged_tree)

    def convert_tree(tree, tokenizer):
        tokenized_tree = []
        for word in tree:
            if isinstance(word, str):
                subword_tokens = tokenizer.tokenize(word)
                if len(subword_tokens) > 1:
                    tokenized_subwords = convert_tree(subword_tokens, tokenizer)
                    tokenized_tree.extend([tokenized_subwords])
                else:
                    tokenized_tree.append(word)
            else:
                tokenized_subtree = convert_tree(word, tokenizer)
                tokenized_tree.append(tokenized_subtree)
        return tokenized_tree

    def tokenize_tree(tree, tokenizer, flag):
        if isinstance(tree, list):
            tokenized_tree = []
            for child in tree:
                if isinstance(child, str):
                    if flag:
                        subwords = tokenizer.tokenize(child.lower())
                    else:
                        subwords = tokenizer.tokenize(' '+child.lower())
                    flag = False
                    subwords = subwords[0] if len(subwords) == 1 else subwords
                    tokenized_tree.append(subwords)
                else:
                    t, flag = tokenize_tree(child, tokenizer, flag) 
                    tokenized_tree.append(t)
            return tokenized_tree, flag
        return tree, flag

    def count_leaves(tree):
        if isinstance(tree, list):
            count = 0
            for child in tree:
                count += count_leaves(child)
            return count
        else:
            return 1

    def double_check_detokenize(sen):
        words = sen.split(' ')
        for i in range(1, len(words)):
            if words[i] in NEG_POSS:
                words[i-1] = words[i-1] + words[i]
                words[i] = ''
        sen = ' '.join(words)
        words = sen.strip().split(' ')
        for i in range(len(words)):
            if words[i].lower() == 'cannot':
                words[i] = 'can'
                words.insert(i+1, 'not')
            elif words[i].lower() == 'gonna':
                words[i] = 'gon'
                words.insert(i+1, 'na')
        sen = ' '.join(words)
        
        return sen.strip()


    sens_idx = []
    sens = []
    trees = []
    nltk_trees = []
    for file_id_i in file_ids:
        sentences = ptb.parsed_sents(file_id_i)
        for sen_tree in sentences:
            
            # get a sentence in a list form from an nltk tree (after ptb detokenization and subword tokenization)
            sen = self.detok.detokenize(filter_words(sen_tree))
            sen = double_check_detokenize(sen)
            sens.append(sen)
            sens_idx.append(self.tokenizer.encode(sen, add_special_tokens=False))
            
            # get a constituent tree in a list form from an nltk tree (after ptb detokenization and subword tokenization)
            try:
                list_tree = tree2list(sen_tree) # get a constituent tree in list form (conatining only POS words)
                if isinstance(list_tree, str):
                    list_tree = [list_tree]
                detokenized_tree = merge_strings_in_tree(list_tree, NEG_POSS) # merge negation and possesion and special tokens 
                tokenized_tree, _ = tokenize_tree(detokenized_tree, self.tokenizer, flag=True) # subword tokenize the tree
                assert count_leaves(tokenized_tree) == len(sens_idx[-1]), f'number of tokens in the tree and the sentence are different, {count_leaves(tokenized_tree)}, {len(sens_idx[-1])}, {sen}'
            except Exception as e:
                print("sen problem: ", sen)
                raise Exception(f'Error in merge_strings_in_tree {e}')
            trees.append(tokenized_tree)
            nltk_trees.append(sen_tree)
    return sens_idx, sens, trees, nltk_trees