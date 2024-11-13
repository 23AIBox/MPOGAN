import torch
import numpy as np
import torch.optim as optim
from models import Generator, Discriminator
import numpy as np
import warnings
from transformers import logging
import MyUtils
import argparse
from params import CUDA, MAX_SEQ_LEN, VOCAB_SIZE, GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, NUM_PG_BATCHES

warnings.filterwarnings('ignore')
logging.set_verbosity_error()


def _get_amps_fits_scorecutoff(data, scores, sc_cutoff) -> list:
    '''
    Get the sequences that meet the screening criteria
    data: List of sequences
    scores: List of scores
    sc_cutoff: Cutoff value of the score
    '''
    sorted_data_scores = sorted(zip(data, scores),
                                key=lambda x: x[1],
                                reverse=True)
    new_data = [x[0] for x in sorted_data_scores if x[1] >= sc_cutoff]
    return new_data


def _get_CDHIT_seqs(origin_seqs, tmp_dir, similarity=0.4) -> list:
    '''
    De-redundancy using CD-HIT
    origin_seqs: List of sequences
    tmp_dir: Temporary directory
    similarity: CDHIT similarity
    '''
    fasta_file = tmp_dir + 'f0.fasta'
    fasta_file_out = tmp_dir + 'f1.fasta'
    MyUtils.seqs_to_file(origin_seqs, fasta_file)
    MyUtils.CD_HIT(input_fasta_file=fasta_file,
                   output_fasta_file=fasta_file_out,
                   similarity=similarity)
    new_seqs = MyUtils.fasta_to_seqs(fasta_file_out)
    return new_seqs


def model_embedded_screening(gen, sample_nums: int, tmp_dir,
                             screen_logs: dict):
    '''
    Model-embedded screening process
    gen: Generator
    sample_nums: The number of samples to be generated
    tmp_dir: Temporary directory
    screen_logs: Record the number of sequences at each step of the screening process
    '''
    with torch.no_grad():
        # Generate samples
        samples = gen.sample(sample_nums).cpu().numpy()
        seqs = [
            MyUtils.vector_to_sequence(sample).replace("0", "")
            for sample in samples
        ]
        seqs = MyUtils.remove_none_seqs(seqs)
        gen_sample_size = len(seqs)
        print(f'Number of generated samples: {gen_sample_size}')

        # Screening by antimicrobial activity predictor
        scores = MyUtils.predict_by_AMP_classifier(seqs)
        seqs = _get_amps_fits_scorecutoff(seqs, scores, AMP_THRESHOLD)
        after_antimic_size = len(seqs)
        print(f'After antimicrobial activity predictor: {after_antimic_size}')

        # Screening by cytotoxicity predictor
        if WITH_TOXIN == 1:
            seqs = MyUtils.toxin_predictor(seqs,
                                           tmp_dir,
                                           threshold=TOXIN_THRESHOLD)
        after_toxin_size = len(seqs)
        print(f'After cytotoxicity predictor: {after_toxin_size}')

        # De-redundancy
        if WITH_CDHIT == 1:
            seqs = _get_CDHIT_seqs(seqs, tmp_dir, CDHIT_THRESHOLD)
        after_cdhit_size = len(seqs)
        print(f'After de-redudancy: {after_cdhit_size}')

    if len(seqs) == 0:  # If no sequences meet the screening criteria
        print(
            'No sequences meet the screening criteria, doubling the number of samples.'
        )
        sample_nums *= 2

        # Recursive call
        seqs, sample_nums, screen_logs = model_embedded_screening(
            gen, sample_nums, tmp_dir, screen_logs)

    screen_logs['gen_sample_size'].append(gen_sample_size)
    screen_logs['after_antimic_size'].append(after_antimic_size)
    screen_logs['after_toxin_size'].append(after_toxin_size)
    screen_logs['after_cdhit_size'].append(after_cdhit_size)

    # seqs: The sequences that meet the screening criteria
    # sample_nums: The number of samples to be generated
    # screen_logs: Record the number of sequences at each step of the screening process
    return seqs, sample_nums, screen_logs


def RTKU(gen, dataset, epoch_labels, rg_labels, epoch, positive_rates: list,
         sample_nums: int, tmp_dir: str, dataset_seq, screen_logs: dict):
    gen.eval()
    dataset = [data.tolist() for data in dataset]

    # Model-embedded screening process
    high_score_seqs, sample_nums, screen_logs = model_embedded_screening(
        gen, sample_nums, tmp_dir, screen_logs)

    good_size = len(high_score_seqs)
    good_seqs_rate = good_size / sample_nums
    positive_rates.append(good_seqs_rate)

    # ================== Real-Time Knowledge-Updating ==================
    dataset_seq = np.array(dataset_seq)
    high_score_seqs = np.array(high_score_seqs)
    max_update_nums = int(UPDATE_RATE * DATASET_MAX_SIZE)
    expect_good_size = int(max_update_nums * UPDATE_GOOD_RATE)

    if good_size <= expect_good_size:
        real_size = max_update_nums - good_size
    else:
        good_size = expect_good_size
        real_size = max_update_nums - expect_good_size

    random_idx = np.random.choice(len(dataset_seq),
                                  size=(real_size, ),
                                  replace=False)
    real_seqs = dataset_seq[random_idx]
    good_seqs = high_score_seqs[:good_size]
    update_seqs = np.concatenate((real_seqs, good_seqs)).tolist()
    update_rg_labels = np.concatenate(
        (np.zeros(real_size), np.ones(good_size))).astype(int)

    assert len(
        update_seqs
    ) == max_update_nums, 'The number of sequences to be updated is incorrect'

    # Dynamic adjustment of the number of samples
    sample_nums = int((max_update_nums / good_seqs_rate) * UPDATE_GOOD_RATE)

    update_nums = len(update_seqs)
    print('Number of sequences to be update: ' + str(update_nums))

    # First-in-first-out
    # Delete the oldest sequences in the dynamic dataset
    to_remove = np.argsort(epoch_labels)[:update_nums]
    dataset = [data for i, data in enumerate(dataset) if i not in to_remove]
    epoch_labels = np.delete(epoch_labels, to_remove)
    rg_labels = np.delete(rg_labels, to_remove)

    # Add new sequences to the dynamic dataset
    new_datas = [MyUtils.sequence_to_vector(seq) for seq in update_seqs]
    dataset += new_datas
    assert len(dataset) == DATASET_MAX_SIZE
    epoch_labels = np.concatenate(
        [epoch_labels, np.repeat(epoch, update_nums)])
    rg_labels = np.concatenate([rg_labels, update_rg_labels])

    # Random shuffle
    assert len(dataset) == len(epoch_labels)
    assert len(dataset) == len(rg_labels)
    perm = np.random.permutation(len(dataset))
    dataset = [dataset[i] for i in perm]
    if CUDA:
        dataset = torch.Tensor(dataset).type(torch.LongTensor)
        dataset = dataset.to(DEVICE)
    else:
        dataset = torch.IntTensor(dataset).type(torch.LongTensor)
    epoch_labels = epoch_labels[perm]
    rg_labels = rg_labels[perm]

    gen.train()

    return dataset, epoch_labels, rg_labels, sample_nums, positive_rates, screen_logs


def fine_tuning(dataset_file, pretrained_gen_model, pretrained_dis_model):

    epoch_labels = np.zeros(
        DATASET_MAX_SIZE)  # epoch labels for dynamic dataset
    rg_labels = np.zeros(
        DATASET_MAX_SIZE,
        dtype=int)  # real/generated labels, 0:real, 1:generated

    # Load dataset
    dataset_seq = MyUtils.get_fasta_dataset(dataset_file)

    # Transform sequences to vectors
    dataset_vec = [
        MyUtils.sequence_to_vector(seq)
        for seq in dataset_seq[:DATASET_MAX_SIZE]
    ]

    sample_nums = SAMPLE_NUMS  # Initial sample number

    # Initialize the generator and discriminator
    gen = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN,
                    DEVICE)
    dis = Discriminator(device=DEVICE, max_len=MAX_SEQ_LEN + 2)

    # Load pre-trained MPOGAN generator and dicriminator
    gen.load_state_dict(torch.load(pretrained_gen_model, map_location=DEVICE))
    dis.load_state_dict(torch.load(pretrained_dis_model, map_location=DEVICE))

    if CUDA:
        gen = gen.to(DEVICE)
        dis = dis.to(DEVICE)
        oracle_samples = torch.Tensor(dataset_vec).type(torch.LongTensor)
        oracle_samples = oracle_samples.to(DEVICE)
    else:
        oracle_samples = torch.IntTensor(dataset_vec).type(torch.LongTensor)

    # Initialize optimizers
    gen_optimizer = optim.Adam(gen.parameters(), lr=GEN_LR)
    dis_optimizer = optim.Adam(dis.parameters(), lr=DIS_LR)

    # Start adversarial fine-tuning
    print('\nStarting Adversarial Fine-tuning...')

    # Record training information
    loss_pg, loss_d, train_acc, positive_rates = [], [], [], []
    screen_logs = {
        'gen_sample_size': [],
        'after_antimic_size': [],
        'after_toxin_size': [],
        'after_cdhit_size': []
    }
    dataset_comp_logs = {'generated_size': [], 'real_size': []}

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nIteration %d\n--------' % (epoch + 1))

        # Real-Time Knowledge-Updating
        oracle_samples, epoch_labels, rg_labels, sample_nums, positive_rates, screen_logs = RTKU(
            gen=gen,
            dataset=oracle_samples,
            epoch_labels=epoch_labels,
            rg_labels=rg_labels,
            epoch=epoch + 1,
            positive_rates=positive_rates,
            sample_nums=sample_nums,
            tmp_dir=tmp_dir,
            dataset_seq=dataset_seq,
            screen_logs=screen_logs)

        # Record dataset composition logs
        dataset_comp_logs['generated_size'].append(
            len(rg_labels[rg_labels == 1]))
        dataset_comp_logs['real_size'].append(len(rg_labels[rg_labels == 0]))

        # Train the generator
        print('\nAdversarial Training Generator:')
        pg_loss = MyUtils.train_generator_PG(gen, gen_optimizer, dis,
                                             NUM_PG_BATCHES, DATASET_MAX_SIZE,
                                             DEVICE)
        loss_pg.append(pg_loss)

        # Train the discriminator
        print('\nAdversarial Training Discriminator:')
        dis_loss, dis_acc = MyUtils.train_discriminator(
            dis,
            dis_optimizer,
            oracle_samples,
            gen,
            d_steps=1,
            epochs=ADV_D_EPOCHS,
            dataset_size=DATASET_MAX_SIZE,
            device=DEVICE)

        loss_d += dis_loss
        train_acc += dis_acc

        # Plot logs
        MyUtils.plot_png([loss_pg], ['PG'], logs_dir + 'ADV_pg_losses.png')
        MyUtils.plot_png([loss_d], ['D'], logs_dir + 'ADV_d_losses.png')
        MyUtils.plot_png([positive_rates], [f'>={AMP_THRESHOLD}'],
                         logs_dir + 'ADV_pos_rates.png')
        MyUtils.plot_png([train_acc], ['train_acc'],
                         logs_dir + 'train_acc.png')
        MyUtils.plot_png([
            screen_logs['gen_sample_size'], screen_logs['after_antimic_size'],
            screen_logs['after_toxin_size'], screen_logs['after_cdhit_size']
        ], [
            'gen_sample_size', 'after_antimic_size', 'after_toxin_size',
            'after_cdhit_size'
        ], logs_dir + 'screen_logs.png')
        MyUtils.plot_png([
            dataset_comp_logs['generated_size'], dataset_comp_logs['real_size']
        ], ['generated_size', 'real_size'], logs_dir + 'dataset_comp_logs.png')

        # Save models
        if epoch % 20 == 19:  # Save models every 20 epochs
            torch.save(
                gen.state_dict(),
                models_dir + run_dir + '{}_'.format(epoch + 1) + 'gen.pth')
            torch.save(
                dis.state_dict(),
                models_dir + run_dir + '{}_'.format(epoch + 1) + 'dis.pth')


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
    pretrained_gen_model = './models/gen_models/adversarial_learning/2000_gen.pth'
    pretrained_dis_model = './models/AMP_classifier/best_model/best_model.pth'

    # Clean run directory (if exists)
    MyUtils.clean_run_dir(run_name)

    # Create run directory
    run_dir = MyUtils.create_run_dir(run_name)
    models_dir = './models/gen_models/'
    logs_dir = './logs/' + run_dir
    tmp_dir = './data/tmp/run_tmp/' + run_dir

    # MPOGAN fine-tuning
    fine_tuning(dataset_file, pretrained_gen_model, pretrained_dis_model)
