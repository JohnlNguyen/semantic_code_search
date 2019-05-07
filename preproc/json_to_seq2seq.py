# coding=utf-8

from __future__ import print_function
import sys, json


def to_seq2seq(json_file, seq_input, seq_output):
    dataset = json.load(open(json_file))
    with open(seq_input, 'w') as f_inp, open(seq_output, 'w') as f_out:
        for example in dataset:
            f_inp.write(' '.join(example['intent_tokens']) + '\n')
            f_out.write(' '.join(example['snippet_tokens']) + '\n')


if __name__ == '__main__':
    json_file = sys.argv[1]
    seq_input = sys.argv[2]
    seq_output = sys.argv[3]
    to_seq2seq(json_file, seq_input, seq_output)
