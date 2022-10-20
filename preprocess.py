import argparse
import collections
import logging
import os
import re
import pickle

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary

SPACE_NORMALIZER = re.compile("\s+")


def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--pretrain-prefix', default='data/demo_data', help='train file prefix')
    parser.add_argument('--train-prefix', default='data/demo_data', help='train file prefix')
    parser.add_argument('--valid-prefix', default='data/demo_data', help='valid file prefix')
    parser.add_argument('--test-prefix', default='data/demo_data',  help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin/how2mcls', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold', default=0, type=int, help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words', default=-1, type=int, help='number of source words to retain')

    return parser.parse_args()


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    
#     a_dict = build_dictionary(['../how2-dataset/how/text/sum_asr_text.txt'])
#     a_dict.finalize(threshold=args.threshold, num_words=args.num_words)
#     a_dict.save(os.path.join(args.dest_dir, 'a_dict.all'))
#     logging.info('Built a dictionary with {} words'.format(len(a_dict)))
    
    all_en_dict = Dictionary.load(os.path.join(args.dest_dir, 'dict.{}'.format('all_en')))
    src_pt_dict = Dictionary.load(os.path.join(args.dest_dir, 'dict.{}'.format('src_pt')))
    logging.info('Loaded a all EN dictionary with {} words'.format(len(all_en_dict)))
    logging.info('Loaded a source PT dictionary with {} words'.format(len(src_pt_dict)))
    
    if args.pretrain_prefix is not None:
        make_binary_dataset(args.pretrain_prefix + '/pretrain_art_en.txt', os.path.join(args.dest_dir, 'pretrain_en.' + 'tran'), all_en_dict)
        make_binary_dataset(args.pretrain_prefix + '/pretrain_sum.txt', os.path.join(args.dest_dir, 'pretrain.' + 'desc'), all_en_dict)
    if args.train_prefix is not None:
        make_binary_dataset(args.train_prefix + '/train_art_en.txt', os.path.join(args.dest_dir, 'train_en.' + 'tran'), all_en_dict)
        make_binary_dataset(args.train_prefix + '/train_art_pt.txt', os.path.join(args.dest_dir, 'train_pt.' + 'tran'), src_pt_dict)
        make_binary_dataset(args.train_prefix + '/train_sum.txt', os.path.join(args.dest_dir, 'train.' + 'desc'), all_en_dict)
    if args.valid_prefix is not None:
            make_binary_dataset(args.valid_prefix + '/val_art_en.txt', os.path.join(args.dest_dir, 'val_en.' + 'tran'), all_en_dict)
            make_binary_dataset(args.valid_prefix + '/val_art_pt.txt', os.path.join(args.dest_dir, 'val_pt.' + 'tran'), src_pt_dict)
            make_binary_dataset(args.valid_prefix + '/val_sum.txt', os.path.join(args.dest_dir, 'val.' + 'desc'), all_en_dict)
    if args.test_prefix is not None:
        make_binary_dataset(args.test_prefix + '/test_art_en.txt', os.path.join(args.dest_dir, 'test_en.' + 'tran'), all_en_dict)
        make_binary_dataset(args.test_prefix + '/test_art_pt.txt', os.path.join(args.dest_dir, 'test_pt.' + 'tran'), src_pt_dict)
        make_binary_dataset(args.test_prefix + '/test_sum.txt', os.path.join(args.dest_dir, 'test.' + 'desc'), all_en_dict)
   
def build_dictionary(filenames, tokenize=word_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                for symbol in word_tokenize(line.strip()):
                    dictionary.add_word(symbol)
                dictionary.add_word(dictionary.eos_word)
    return dictionary

def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()
    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


if __name__ == '__main__':
    args = get_args()
    utils.init_logging(args)
    main(args)
