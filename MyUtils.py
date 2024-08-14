'''
This file contains utility functions for MPOGAN model training and sequence analysis.
'''

import os
import re
import csv
import random
import sys
import shutil
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Bio import SeqIO

import modlamp.descriptors
import modlamp.analysis
import modlamp.sequences
import toxinpred2
from models import Generator, Discriminator

from params import (
    CHARACTER_DICT, 
    INDEX_DICT, 
    SEQ_LEN, 
    MAX_SEQ_LEN, 
    VOCAB_SIZE, 
    START_LETTER, 
    BATCH_SIZE, 
    GEN_EMBEDDING_DIM, 
    GEN_HIDDEN_DIM
)


def clean_run_dir(run_name):
    """
    Function to clean run directory by deleting model and log folders.
    """
    model_dir = f"./models/gen_models/{run_name}"
    logs_dir = f"./logs/{run_name}"

    if run_name != "":
        if os.path.exists(model_dir):
            print("Deleting model directory")
            shutil.rmtree(model_dir)
        else:
            print("Model directory does not exist")

        if os.path.exists(logs_dir):
            print("Deleting logs directory")
            shutil.rmtree(logs_dir)
        else:
            print("Logs directory does not exist")
    else:
        raise Exception("Please set run_name")


def create_dataset(seqs_len, file):
    '''Create dataset from csv file'''
    all_seqs = []
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            seq = row[0]
            length = len(seq)
            remove = False
            for aa in seq:
                if aa not in [
                        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                        'Y', 'Z'
                ]:
                    remove = True
                    break
            if remove:
                continue
            if length >= seqs_len[0] and length <= seqs_len[1] and row[
                    2] == 'Experimentally validated':
                all_seqs.append(row[0])
    print('Obtained {} sequences.'.format(len(all_seqs)))
    return all_seqs


def fasta_to_seqs(fastafile):
    '''Read fasta file and return sequences'''
    seqs = []
    for seq in SeqIO.parse(fastafile, 'fasta'):
        seqs.append(str(seq.seq))
    return seqs


def seqs_to_file(seqs, file):
    '''Write sequences to file'''
    with open(file, 'w') as f:
        for i, seq in enumerate(seqs):
            f.write('>seq{}\n{}\n'.format(i, seq))


def remove_abnormal_aas(all_sequences: list):
    '''Delete sequences containing amino acids: X, Z, U, O'''
    all_sequences = [
        seq for seq in all_sequences if ('X' not in seq) and (
            'Z' not in seq) and ('U' not in seq) and ('O' not in seq)
    ]
    return all_sequences


def remove_none_seqs(seqs: list):
    '''Remove empty sequences'''
    seqs = [seq for seq in seqs if len(seq) > 0]
    return seqs


def get_fasta_dataset(file_path):
    '''Load fasta dataset'''
    all_sequences = fasta_to_seqs(file_path)
    all_sequences = remove_abnormal_aas(all_sequences)
    return all_sequences


def get_csv_dataset(file_path):
    '''Load csv dataset'''
    df = pd.read_csv(file_path)
    df = df[df.Sequence.str.len().between(SEQ_LEN[0], SEQ_LEN[1])]  # 限制序列长度
    all_sequences = df.Sequence.to_numpy().tolist()
    all_sequences = remove_abnormal_aas(all_sequences)
    return all_sequences


def sequence_to_vector(sequence):
    '''Encode sequence to integer vector'''
    default = np.asarray([21] * (MAX_SEQ_LEN))
    for i, character in enumerate(sequence[:MAX_SEQ_LEN]):
        try:
            default[i] = CHARACTER_DICT[character]
        except KeyError:
            continue
    return default.astype(int)


def vector_to_sequence(vector):
    '''Decode integer vector to sequence'''
    return ''.join([INDEX_DICT.get(item, '0') for item in vector])


def write_losses(losses, file):
    '''Write losses to file'''
    with open(file, "w") as f:
        for loss in losses:
            f.write("{}\n".format(loss))


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_png(losses_list,
             legends_list,
             file_out=None,
             alpha=0.2,
             is_smooth=True,
             save_csv=True):
    '''Plot losses'''
    assert len(losses_list) == len(legends_list)
    plt.figure(figsize=(5, 5))
    colors = [
        '#FF0000',
        '#0000FF',
        '#00EE76',
        '#9400D3',
        '#8B7E66',
        '#00FFFF',
        '#5F9EA0',
        '#BDB76B',
    ]
    for i, loss in enumerate(losses_list):
        if is_smooth:
            smoothed_loss = smooth_curve(loss)
            plt.plot(loss, alpha=alpha, color=colors[i])
            plt.plot(smoothed_loss,
                     label=legends_list[i],
                     alpha=1.0,
                     color=colors[i])
        else:
            plt.plot(loss, alpha=1.0, color=colors[i], label=legends_list[i])

    plt.legend()
    if file_out is None:
        plt.show()
    else:
        plt.savefig(file_out)
    plt.close()

    if save_csv:
        df = pd.DataFrame(
            dict([(k, pd.Series(v))
                  for k, v in zip(legends_list, losses_list)]))
        csv_out = file_out.replace('.png', '.csv')
        df.to_csv(csv_out, header=True, index_label='index')


def create_run_dir(run_name: str = "default_run"):
    '''Create run directory'''

    if run_name == "":
        raise ValueError("run_name字段不能为空")

    if os.path.exists('./logs/' + run_name):
        os.system('rm -rf ./logs/' + run_name)
    if os.path.exists('./models/gen_models/' + run_name):
        os.system('rm -rf ./models/gen_models/' + run_name)
    if os.path.exists(f'./data/tmp/run_tmp/{run_name}'):
        os.system(f'rm -rf ./data/tmp/run_tmp/{run_name}')

    os.mkdir('./logs/' + run_name)
    os.mkdir('./models/gen_models/' + run_name)
    os.mkdir(f'./data/tmp/run_tmp/{run_name}')

    return run_name + '/'

def generate_seqs(run_name,
                  model_name,
                  sample_nums,
                  device: str = 'cpu',
                  max_seq_len=MAX_SEQ_LEN):
    '''Generate sequences using MPOGAN generator'''

    gen_model = f"./models/gen_models/{run_name}/{model_name}"
    gen = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, max_seq_len,
                    device)

    gen.load_state_dict(
        torch.load(gen_model, map_location=torch.device(device)))

    gen.eval()
    a = gen.sample(sample_nums)
    a = a.tolist()
    seqs = []
    for i in range(sample_nums):
        seq = vector_to_sequence(a[i])
        seq = re.sub('[X]+$', '', seq)
        seq = seq.replace("0", "")
        seqs.append(seq)

    return seqs
