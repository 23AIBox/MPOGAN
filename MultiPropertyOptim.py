import torch
import numpy as np
import torch.optim as optim
from models import Generator, Discriminator
import numpy as np
from transformers import logging
import MyUtils
import argparse
from params import CUDA, MAX_SEQ_LEN, VOCAB_SIZE, GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, NUM_PG_BATCHES

def fine_tuning(dataset_file, pretrained_gen_model, pretrained_dis_model):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FB-SeqGAN模型训练')
    parser.add_argument('--run_name',
                        type=str,
                        default='MPOGAN_finetuning',
                        help='Run name of the task')
    parser.add_argument('--device', type=str, default="cuda:1", help='device')
    parser.add_argument('--dataset_max_size',
                        type=int,
                        default=1000,
                        help='The maximum size of the dynamic dataset')
    parser.add_argument('--epoch',
                        type=int,
                        default=1000,
                        help='Fine-tuning epochs')
    parser.add_argument('--gen_lr',
                        type=float,
                        default=0.00005,
                        help='Leaning rate of the generator')
    parser.add_argument('--dis_lr',
                        type=float,
                        default=0.00005,
                        help='Leaning rate of the discriminator')
    parser.add_argument('--ur',
                        type=float,
                        default=0.25,
                        help='Update rate of the dynamic dataset')
    parser.add_argument(
        '--ugr',
        type=float,
        default=0.7,
        help=
        'The proportion of generated sequences in the sequences to be updated')
    parser.add_argument(
        '--with_toxin',
        type=int,
        default=1, # 0: no, 1: yes
        help=
        'Ablaition study: whether to include the cytotoxicity predictor in the model-embedded screening process'
    )
    parser.add_argument(
        '--with_cdhit',
        type=int,
        default=1, # 0: no, 1: yes
        help=
        'Ablaition study: whether to include de-redundancy module in the model-embedded screening process'
    )
    parser.add_argument('--amp_threshold',
                        type=float,
                        default=0.8,
                        help='Cutoff value of AMP predictor scores')
    parser.add_argument('--toxin_threshold',
                        type=float,
                        default=0.7,
                        help='Cutoff value of cytotoxicity predictor scores')
    parser.add_argument('--cdhit_threshold',
                        type=float,
                        default=0.6,
                        help='Cutoff value of CDHIT similarity')
    args = parser.parse_args()

    torch.manual_seed(2024)  # Set random seed
    DEVICE = args.device

    run_name = args.run_name  # Run name of the task

    DATASET_MAX_SIZE = args.dataset_max_size  # The maximum size of the dynamic dataset
    UPDATE_RATE = args.ur  # Update rate of the dynamic dataset
    UPDATE_GOOD_RATE = args.ugr  # The proportion of generated sequences in the sequences to be updated

    ADV_TRAIN_EPOCHS = args.epoch  # Fine-tuning epochs
    SAMPLE_NUMS = 3000  # Initial sample number

    GEN_LR = args.gen_lr  # Learning rate of the generator
    DIS_LR = args.dis_lr  # Learning rate of the discriminator

    ADV_D_EPOCHS = 5  # Discriminator training epochs

    AMP_THRESHOLD = args.amp_threshold  # Cutoff value of AMP predictor scores
    TOXIN_THRESHOLD = args.toxin_threshold  # Cutoff value of cytotoxicity predictor scores
    CDHIT_THRESHOLD = args.cdhit_threshold  # Cutoff value of CDHIT similarity

    # Ablaition study
    WITH_TOXIN = args.with_toxin
    WITH_CDHIT = args.with_cdhit

    # Load dataset
    dataset_file = './data/positive.fasta'

    # Load pre-trained MPOGAN generator and dicriminator
    pretrained_gen_model = ''
    pretrained_dis_model = ''

    # Clean run directory (if exists)
    MyUtils.clean_run_dir(run_name)

    # Create run directory
    run_dir = MyUtils.create_run_dir(run_name)
    models_dir = './models/gen_models/'
    logs_dir = './logs/' + run_dir
    tmp_dir = './data/tmp/run_tmp/' + run_dir

    # MPOGAN fine-tuning
    fine_tuning(dataset_file, pretrained_gen_model, pretrained_dis_model)
