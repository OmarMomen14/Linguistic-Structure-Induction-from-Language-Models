# Master's Thesis: Linguistic Structure Induction from Language Models

This repository contains the experiment codes for the master's thesis "Linguistic Structure Induction from Language Models".

## Installation

`pip install -r requirements.txt`

## Usage

1. Download PTB constituency annotations data (for constituency evaluation) and place them in the nltk directory at the system root.

2. Download your desired PTB dependency annotations data e.g stanford,conll,conll-u (for dependency evaluation) and place them in the `data/ptb` dir.

3. Scripts and commands:

  	+  Train and test original StructFormer

      ```python -m main.py --model structformer --cuda --pos_emb --save /path/to/your/saved_model --data ./data/penn/ --test_grammar --print```
        
        - For vanilla transformer use `--model transformer`
        - For In-parser StructFormer use `--model structformer_in_parser`
        - For Subword Tokenization use `--subword <tokenizer_path>` e.g `omarmomen/ptb_filtered_lowcase_bpe_tokenizer_8`


  	+ Test Unsupervised Parsing only (For word-tokenizer)
    
    ```python -m test_phrase_grammar.py --cuda --checkpoint /path/to/your/model --print```
    
        - For Subword Tokenization use `--subword <tokenizer_path>` e.g `omarmomen/ptb_filtered_lowcase_bpe_tokenizer_8`

## Acknowledgements

Much of the project code is based on the following repository:  

- [StructFormer](https://github.com/google-research/google-research/tree/master/structformer) 
- [BabyLM Eval](https://github.com/babylm/evaluation-pipeline) 