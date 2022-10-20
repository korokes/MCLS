import argparse
import math
import logging
import os
import random
import torch
import torch.nn as nn
from torch.serialization import default_restore_location

import sys
lib_path = os.path.abspath('./')
sys.path.append(lib_path)
from transformers import transformers

from itertools import chain
from tqdm import tqdm
from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, How2MCLS_Dataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


def get_args():
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--seed', default=16, type=int, help='pseudo random number generator seed')
    parser.add_argument('--distributed-world-size', default=torch.cuda.device_count(), help='distributed world size')
    parser.add_argument('--distributed-backend', default='nccl', help='distributed backend')

    # Add data arguments
    parser.add_argument('--data', default='data-bin/how2mcls', help='path to data directory') #data-bin/how2mcls
    parser.add_argument('--max-tokens', default=16000, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--train_video_file', default='data/demo_data/train_action.txt', help='name of train video file') 
    parser.add_argument('--val_video_file', default='data/demo_data/val_action.txt', help='name of val video file') 
    parser.add_argument('--video_dir', default='data/demo_data/video_action_features', help='path of video features')
    parser.add_argument('--batch-size', default=8, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--num-workers', default=0, type=int, help='number of data workers')

    # Add model arguments
    parser.add_argument('--arch', default='DAtransformer', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=50, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=0.1, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.00015, type=float, help='learning rate')
    parser.add_argument('--warmup-steps', default=4000, type=float, help='warmup steps')
    parser.add_argument('--momentum', default=0.99, type=float, help='momentum factor')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--lr-shrink', default=0.1, type=float, help='learning rate shrink factor for annealing')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='minimum learning rate')

    # Add checkpoint arguments
    parser.add_argument('--log-file', default='./log_dir/log_s2.txt', help='path to save logs') 
    parser.add_argument('--target-dir', default='checkpoints/model_s1', help='path to save checkpoints') 
    parser.add_argument('--save-dir', default='checkpoints/model_s2', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')
    
    
    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported.')
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device_id)
    utils.init_logging(args)

    if args.distributed_world_size > 1:
        torch.distributed.init_process_group(
            backend=args.distributed_backend, init_method=args.distributed_init_method,
            world_size=args.distributed_world_size, rank=args.distributed_rank)

    # Load dictionaries
    all_en_dict = Dictionary.load(os.path.join(args.data, 'dict.all_en'))
    logging.info('Loaded a EN dictionary with {} words'.format(len(all_en_dict)))
    src_pt_dict = Dictionary.load(os.path.join(args.data, 'dict.src_pt'))
    logging.info('Loaded a PT dictionary with {} words'.format(len(src_pt_dict)))
    

    # Load datasets
    def load_data(split,video_file=None, video_dir=args.video_dir):
          return How2MCLS_Dataset(
              src_en_file=os.path.join(args.data, '{}_en.{}'.format(split, 'tran')),
              src_pt_file=os.path.join(args.data, '{}_pt.{}'.format(split, 'tran')),
              tgt_en_file=os.path.join(args.data, '{}.{}'.format(split, 'desc')),
              all_en_dict=all_en_dict,
              src_pt_dict=src_pt_dict,
              video_file=video_file, 
              video_dir=video_dir)    

    train_dataset = load_data(split='train',video_file=args.train_video_file)
    valid_dataset = load_data(split='val',video_file=args.val_video_file)

    # Build model and criterion
    model = models.build_model(args, all_en_dict, src_pt_dict).cuda()
    logging.info('Built a model with {} parameters'.format(sum(p.numel() for p in model.parameters())))
    
    # Load last checkpoint if one exists
    def load_source_checkpoint(args, model):
        checkpoint_path = os.path.join(args.target_dir, args.restore_file)
        if os.path.isfile(checkpoint_path):
            target_model = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            model_dict = model.state_dict()
            state_dict = {'target_'+k:v for k,v in target_model['model'].items() if 'target_'+k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            logging.info('Loaded checkpoint {}'.format(checkpoint_path))
            return state_dict

    load_source_checkpoint(args, model)
    last_epoch =  -1

    print('model parameters:')
    count = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        print(name)
        count += 1
    print('count of all grad name: {}'.format(count))

    print('trainable parameters:')
    count = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            print(name)
            count +=1 
    print('count of requires grad name: {}'.format(count))
    
    #criterion = nn.CrossEntropyLoss(ignore_index=all_en_dict.pad_idx, size_average=False).cuda()
    criterion = torch.nn.SmoothL1Loss().cuda()

    # Build an optimizer and a learning rate schedule
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True, betas=(0.9, 0.98))
    optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, correct_bias=True)
    with open(args.train_video_file, 'r', encoding='utf8') as f:
        data_size = len([line.strip() for line in f])
    total_steps = int(data_size * args.max_epoch / args.batch_size) 
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                                          t_total=total_steps) 

    for epoch in range(last_epoch + 1, args.max_epoch):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=args.num_workers, collate_fn=train_dataset.collater,
            batch_sampler=BatchSampler(
                train_dataset, args.max_tokens, args.batch_size, args.distributed_world_size,
                args.distributed_rank, shuffle=True, seed=args.seed))
        
        model.train()
        stats = {'loss': 0., 'lr': 0., 'num_tokens': 0., 'batch_size': 0., 'grad_norm': 0., 'clip': 0.}
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=(args.distributed_rank != 0))

        for i, sample in enumerate(progress_bar):

            sample = utils.move_to_cuda(sample)

            if len(sample) == 0:
                continue

            # Forward and backward pass
            label, output = model(sample['src_en_tokens'], sample['src_en_lengths'], sample['src_pt_tokens'], sample['src_pt_lengths']) 
            loss = criterion(output, label)
            #loss = criterion(output.reshape(-1, output.size(-1)), label.reshape(-1, label.size(-1)))
            optimizer.zero_grad()
            loss.backward()

            # Reduce gradients across all GPUs
            if args.distributed_world_size > 1:
                utils.reduce_grads(model.parameters())
                total_loss, num_tokens, batch_size = list(map(sum, zip(*utils.all_gather_list(
                    [loss.item(), sample['num_tokens'], len(sample['src_pt_tokens'])]))))
            else:
                total_loss, num_tokens, batch_size = loss.item(), sample['num_tokens'], len(sample['src_pt_tokens'])

            # Normalize gradients by number of tokens and perform clipping
            for param in model.parameters():
                if param.requires_grad:
                    param.grad.data.div_(num_tokens)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            scheduler.step()

            # Update statistics for progress bar
            stats['loss'] += total_loss / num_tokens / math.log(2)
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += num_tokens / len(sample['src_pt_tokens'])
            stats['batch_size'] += batch_size
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()}, refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))

        # Adjust learning rate based on validation loss
        valid_loss = validate(args, model, criterion, valid_dataset, epoch)

        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, epoch, valid_loss)
        if optimizer.param_groups[0]['lr'] <= args.min_lr:
            logging.info('Done training!')
            break


def validate(args, model, criterion, valid_dataset, epoch):
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, num_workers=args.num_workers, collate_fn=valid_dataset.collater,
        batch_sampler=BatchSampler(
            valid_dataset, args.max_tokens, args.batch_size, args.distributed_world_size,
            args.distributed_rank, shuffle=True, seed=args.seed))

    model.eval()
    stats = {'valid_loss': 0, 'num_tokens': 0, 'batch_size': 0}
    progress_bar = tqdm(valid_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

    for i, sample in enumerate(progress_bar):
        sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue
        with torch.no_grad():
            label, output = model(sample['src_en_tokens'], sample['src_en_lengths'], sample['src_pt_tokens'], sample['src_pt_lengths'])
            loss = criterion(output, label)
        stats['valid_loss'] += loss.item() / sample['num_tokens'] / math.log(2)
        stats['num_tokens'] += sample['num_tokens'] / len(sample['src_pt_tokens'])
        stats['batch_size'] += len(sample['src_pt_tokens'])
        progress_bar.set_postfix({key: '{:.3g}'.format(value / (i + 1)) for key, value in stats.items()}, refresh=True)

    logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(
        value / len(progress_bar)) for key, value in stats.items())))
    return stats['valid_loss'] / len(progress_bar)


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = get_args()
    if args.distributed_world_size == 1:
        args.distributed_rank = 0
        args.device_id = 0
        main(args)
    else:
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=random.randint(10000, 20000))
        mp = torch.multiprocessing.get_context('spawn')
        processes = []
        for rank in range(args.distributed_world_size):
            args.device_id = rank
            args.distributed_rank = rank
            processes.append(mp.Process(target=main, args=(args,)))
            processes[rank].start()
        for process in processes:
            process.join()
