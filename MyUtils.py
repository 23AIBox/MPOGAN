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
    df = df[df.Sequence.str.len().between(SEQ_LEN[0], SEQ_LEN[1])]
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
        raise ValueError("run_name cannot be none")

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


def prepare_generator_batch(samples, device: str, start_letter=START_LETTER):
    '''Prepare batch for generator training'''
    batch_size, seq_len = samples.size()
    inp = torch.zeros(batch_size, seq_len)
    target = samples
    target_seqs = [
        vector_to_sequence(vec).replace("0", "")
        for vec in target.cpu().numpy()
    ]

    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len - 1]
    inp = inp.type(torch.LongTensor)
    target = target.type(torch.LongTensor)

    inp = inp.to(device)
    target = target.to(device)

    return inp, target, target_seqs


def prepare_discriminator_data(pos_samples, neg_samples, device: str):
    '''Prepare data for discriminator training'''
    inp = torch.cat((pos_samples, neg_samples), 0).long()
    inp_seqs = [
        vector_to_sequence(vec).replace("0", "") for vec in inp.cpu().numpy()
    ]
    target_argmax = torch.cat(
        (torch.ones(pos_samples.size(0)), torch.zeros(neg_samples.size(0))), 0)
    target = torch.nn.functional.one_hot(target_argmax.to(torch.int64),
                                         num_classes=2).float()

    perm = torch.randperm(target.size(0))
    inp_seqs = [inp_seqs[i] for i in perm]
    target = target[perm]
    target_argmax = target_argmax[perm]

    target = target.to(device)

    return inp_seqs, target, target_argmax


def batchwise_sample(gen, num_samples, batch_size):
    '''Sample sequences batchwise'''

    samples = []
    for i in range(int(ceil(num_samples / float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def train_generator_PG(gen, gen_opt, dis, num_batches, dataset_size, device):
    '''Train generator using policy gradient'''

    total_pg_loss = 0

    for _ in range(num_batches):

        s = gen.sample(BATCH_SIZE * 2)
        inp, target, target_seqs = prepare_generator_batch(
            s, start_letter=START_LETTER, device=device)

        dis_outputs = dis(target_seqs)
        rewards = dis_outputs[:, 1]

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

        total_pg_loss += pg_loss.data.item()

    total_pg_loss /= num_batches

    print('Generator PG loss = %.4f' % total_pg_loss)

    return total_pg_loss


def train_discriminator(discriminator, dis_opt, real_data_samples, generator,
                        d_steps, epochs, dataset_size, device: str):
    '''Train discriminator'''
    losses, accs = [], []

    for d_step in range(d_steps):
        print(f"\nd-step {d_step + 1}:\n\tepoch\tloss\ttr_acc\tval_acc")

        indice = random.sample(range(len(real_data_samples)), 100)
        indice = torch.tensor(indice)
        pos_val = real_data_samples[indice]
        neg_val = generator.sample(100)
        val_inp, val_target, val_target_argmax = prepare_discriminator_data(
            pos_val, neg_val, device)

        s = batchwise_sample(generator, dataset_size, BATCH_SIZE)
        dis_inp, dis_target, dis_target_argmax = prepare_discriminator_data(
            real_data_samples, s, device)

        for epoch in range(epochs):
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * dataset_size, BATCH_SIZE):

                inp, target, target_argmax = dis_inp[
                    i:i + BATCH_SIZE], dis_target[
                        i:i + BATCH_SIZE], dis_target_argmax[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator(inp)

                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                out_argmax = torch.max(out, 1)[1].cpu().numpy()
                for j in range(len(out_argmax)):
                    if target_argmax[j] == out_argmax[j]:
                        total_acc += 1

            total_loss /= ceil(2 * dataset_size / float(BATCH_SIZE))
            total_acc /= float(2 * dataset_size)

            val_acc = 0
            val_pred = discriminator(val_inp)
            val_pred_argmax = torch.max(val_pred, 1)[1].cpu().numpy()
            for j in range(len(val_pred_argmax)):
                if val_pred_argmax[j] == val_target_argmax[j]:
                    val_acc += 1
            val_acc /= float(len(val_pred_argmax))

            print('\t%d\t\t%.4f\t%.4f\t%.4f' %
                  (epoch + 1, total_loss, total_acc, val_acc))

            losses.append(total_loss)
            accs.append(total_acc)

    return losses, accs


def predict_by_AMP_classifier(seqs: list,
                              device='cpu',
                              max_len=MAX_SEQ_LEN + 2) -> list:
    '''Predict AMP probability by antimicrobial activity predictor (LLM-AAP)'''

    model_path = "./models/AMP_classifier/best_model/best_model.pth"
    dis = Discriminator(device=device, max_len=max_len)
    dis.load_state_dict(
        torch.load(model_path, map_location=torch.device(device)))
    dis.to(device)
    dis.eval()

    with torch.no_grad():
        output = dis(seqs)
        output = output.cpu().numpy()
        score = output[:, 1].tolist()
    return score


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


def calculate_length(data: list):
    lengths = [len(x) for x in data]
    return lengths


def calculate_molarweight(data: list):
    h = modlamp.descriptors.GlobalDescriptor(data)
    h.calculate_MW()
    return list(h.descriptor.flatten())


def calculate_charge(data: list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_charge()
    return h.charge


def calculate_isoelectricpoint(data: list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return list(h.descriptor.flatten())


def calculate_aromaticity(data: list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.aromaticity()
    return list(h.descriptor.flatten())


def calculate_hydrophobicity(data: list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_H(scale='eisenberg')
    return list(h.H)


def calculate_hydrophobicmoment(data: list):
    h = modlamp.descriptors.PeptideDescriptor(data, 'eisenberg')
    h.calculate_moment()
    return list(h.descriptor.flatten())


def calculate_hydrophobic_ratio(data: list):
    h = modlamp.descriptors.GlobalDescriptor(data)
    h.hydrophobic_ratio()
    return list(h.descriptor.flatten())


def calculate_charge_density(data: list):
    h = modlamp.descriptors.GlobalDescriptor(data)
    h.charge_density()
    return list(h.descriptor.flatten())


def calculate_instability_index(data: list):
    h = modlamp.descriptors.GlobalDescriptor(data)
    h.instability_index()
    return list(h.descriptor.flatten())


def calculate_aliphatic_index(data: list):
    h = modlamp.descriptors.GlobalDescriptor(data)
    h.aliphatic_index()
    return list(h.descriptor.flatten())


def calculate_physchems(peptides: list):
    '''Calculate physicochemical properties of peptides'''
    physchem = {}
    physchem['Charge'] = calculate_charge(peptides)[0].tolist()
    physchem['Isoelectric point'] = calculate_isoelectricpoint(peptides)
    physchem['Aromaticity'] = calculate_aromaticity(peptides)
    physchem['Eisenberg hydrophobicity'] = calculate_hydrophobicity(
        peptides)[0].tolist()
    physchem['Hydrophobic moment'] = calculate_hydrophobicmoment(peptides)
    physchem['Hydrophobic ratio'] = calculate_hydrophobic_ratio(peptides)
    physchem['Charge density'] = calculate_charge_density(peptides)
    physchem['Instability index'] = calculate_instability_index(peptides)
    physchem['Aliphatic index'] = calculate_aliphatic_index(peptides)

    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in physchem.items()]))


def CD_HIT(input_fasta_file, output_fasta_file, similarity=0.4):
    '''De-redundancy of sequences using CD-HIT'''
    os.system(
        f"./models/cd-hit-v4.6.7-2017-0501/cd-hit -i {input_fasta_file} -o {output_fasta_file} -c {similarity} -T 4 -n 2 -M 2000 >/dev/null 2>&1"
    )


def toxin_predictor(
    origin_seqs: list,
    tmp_dir: str,
    threshold: float = 0.5,
) -> list:
    '''Predict cytotoxicity of sequences using ToxinPred2'''

    scs = toxinpred2.toxinpred2(origin_seqs, tmp_dir)
    df = pd.DataFrame(origin_seqs)
    df.columns = ['sequence']
    df['toxin_score'] = scs
    df_good = df[df['toxin_score'] < threshold]

    return df_good['sequence'].tolist()
