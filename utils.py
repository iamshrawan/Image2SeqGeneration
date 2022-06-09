
import random
from statistics import mode
import numpy as np
import os
import glob
import torch

from loss import losses_node


def string_tuple_to_list(string_tuple):
    non_bracket = string_tuple[1:-1]
    sequence = non_bracket.split(',')
    return sequence

def create_vocabulary(labels):
    sequences = list(labels.values())
    unique_symbols = sorted(list(set(sym for seq in sequences for sym in seq)))
    return unique_symbols

def map_sym2idx(sym, uniq):
    return uniq.index(sym)

def map_idx2sym(idx, uniq):
    return uniq[idx]


def train_valid_test_split(num_images, num_test, num_dev):
    all_indices = np.arange(num_images)
    total_set_aside = num_test + num_dev
    set_aside_indices = []
    for i in range(total_set_aside):
        if i==0:
            idx = i+1
        else:
            idx = set_aside_indices[-1] + 2
        set_aside_indices.append(all_indices[idx])
    
    test_indices = random.sample(set_aside_indices, 2)
    dev_indices = list(np.setdiff1d(set_aside_indices, test_indices))
    
    return test_indices, dev_indices

def get_train_dev_test_examples(path, test_indices, dev_indices):
    examples = sorted(glob.glob(os.path.join(path, '*.png')))
    test_examples = [ex for ex in examples if int(os.path.splitext(ex)[0][-1]) in test_indices]
    dev_examples = [ex for ex in examples if int(os.path.splitext(ex)[0][-1]) in dev_indices]
    train_examples = list(np.setdiff1d(examples, test_examples+dev_examples))
    return test_examples, dev_examples, train_examples

def get_validation_metric(model, dataloader, opt):
    model.eval()  
    val_losses = 0
    for batch in dataloader:
        image = batch['image']
        input_op_idx, label, program_lens = batch['inp_op'], batch['label'], batch['program_len']
        # Reshaping and getting one hot encoding of input operations
        input_op = torch.zeros((input_op_idx.shape[0], input_op_idx.shape[1], opt.unq_op+2))
        input_op = input_op.scatter_(2, input_op_idx.unsqueeze(2), 1)

        if opt.cuda:
            image = image.cuda()
            input_op = input_op.cuda()
            program_len = program_lens[-1].cuda()

        output = model([image, input_op, program_len])
        loss = losses_node(out=output, labels=label, time_steps=program_len+1)
        val_losses += loss
    val_losses /= len(dataloader)
    return val_losses
    
def beams_parser(all_beams, batch_size, beam_width=5):
    # all_beams = [all_beams[k].data.numpy() for k in all_beams.keys()]
    all_expression = {}
    W = beam_width
    T = len(all_beams)
    for batch in range(batch_size):
        all_expression[batch] = []
        for w in range(W):
            temp = []
            parent = w
            for t in range(T - 1, -1, -1):
                temp.append(all_beams[t]["index"][batch, parent].data.cpu()
                            .numpy())
                parent = all_beams[t]["parent"][batch, parent]
            temp = temp[::-1]
            all_expression[batch].append(np.array(temp))
        all_expression[batch] = np.squeeze(np.array(all_expression[batch]))
    return all_expression
