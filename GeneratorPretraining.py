import sys
from math import ceil
import torch
import torch.optim as optim
import warnings
from models import Generator
import MyUtils
import argparse
from params import CUDA, MAX_SEQ_LEN, VOCAB_SIZE, START_LETTER, \
    GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM

warnings.filterwarnings('ignore')


def train_generator_MLE(gen,
                        gen_opt,
                        real_data_samples,
                        epochs,
                        logs_dir,
                        models_dir,
                        run_dir,
                        dataset_size,
                        device,
                        save_middle_models=False,
                        is_draw_smooth=True):
    '''Train generator using MLE loss'''
    total_losses, oracle_losses = [], []
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, dataset_size, BATCH_SIZE):
            inp, target, _ = MyUtils.prepare_generator_batch(
                real_data_samples[i:i + BATCH_SIZE],
                start_letter=START_LETTER,
                device=device)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                    ceil(dataset_size / float(BATCH_SIZE)) / 10.) == 0:
                print('.', end='')
                sys.stdout.flush()
        print('done.')

        total_loss = total_loss / ceil(
            dataset_size / float(BATCH_SIZE)) / MAX_SEQ_LEN

        print('NLL_loss = %.4f' % total_loss)

        total_losses.append(total_loss)
        MyUtils.plot_png([total_losses], ['total_lossess'],
                         logs_dir + 'total_lossess.png',
                         is_smooth=is_draw_smooth)

        if epoch % 20 == 0 and save_middle_models:
            torch.save(
                gen.state_dict(),
                models_dir + run_dir + '{}_pretrain_gen.pth'.format(epoch))

    return total_losses, oracle_losses


def train_seqgan(dataset_seq, run_name):
    dataset_vec = [MyUtils.sequence_to_vector(seq) for seq in dataset_seq]
    run_dir = MyUtils.create_run_dir(run_name)
    models_dir = './models/gen_models/'
    logs_dir = './logs/' + run_dir

    gen = Generator(GEN_EMBEDDING_DIM,
                    GEN_HIDDEN_DIM,
                    VOCAB_SIZE,
                    MAX_SEQ_LEN,
                    device=DEVICE)

    if CUDA:
        gen = gen.cuda()
        oracle_samples = torch.Tensor(dataset_vec).type(torch.LongTensor)
        oracle_samples = oracle_samples.cuda()
    else:
        oracle_samples = torch.IntTensor(dataset_vec).type(torch.LongTensor)

    print('\nStarting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=GEN_LR)
    train_generator_MLE(gen,
                        gen_optimizer,
                        oracle_samples,
                        MLE_TRAIN_EPOCHS,
                        logs_dir,
                        models_dir,
                        run_dir,
                        DATASET_MAX_SIZE,
                        device=DEVICE,
                        save_middle_models=True,
                        is_draw_smooth=False)

    torch.save(gen.state_dict(), models_dir + run_dir + 'pre_gen.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generator Pre-training')
    parser.add_argument('--run_name', type=str, help='Name of the experiment')
    args = parser.parse_args()

    dataset_seq = MyUtils.get_fasta_dataset('./data/positive.fasta')
    print(f'Dataset size: {len(dataset_seq)}')

    torch.manual_seed(2024)
    DATASET_MAX_SIZE = len(dataset_seq)
    MLE_TRAIN_EPOCHS = 1000
    GEN_LR = 0.0001
    BATCH_SIZE = 256

    DEVICE = "cuda:0"

    MyUtils.clean_run_dir(args.run_name)
    train_seqgan(dataset_seq, args.run_name)
