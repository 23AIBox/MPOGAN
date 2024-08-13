import torch
import os
import torch.optim as optim
from models import Generator, Discriminator
import MyUtils
import argparse
from params import CUDA, MAX_SEQ_LEN, VOCAB_SIZE, GEN_EMBEDDING_DIM,\
    GEN_HIDDEN_DIM, NUM_PG_BATCHES


def adversarial_learning(dataset_seq, run_name, pretrained_gen,
                         pretrained_dis):

    dataset_vec = [MyUtils.sequence_to_vector(seq) for seq in dataset_seq]

    run_dir = MyUtils.create_run_dir(run_name)
    models_dir = './models/gen_models/'
    logs_dir = './logs/' + run_dir

    gen = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN,
                    DEVICE)
    dis = Discriminator(DEVICE, MAX_SEQ_LEN + 2)

    if CUDA:
        gen = gen.to(DEVICE)
        dis = dis.to(DEVICE)
        real_samples = torch.Tensor(dataset_vec).type(torch.LongTensor)
        real_samples = real_samples.to(DEVICE)
    else:
        real_samples = torch.IntTensor(dataset_vec).type(torch.LongTensor)

    # Define the optimizers
    gen_optimizer = optim.Adam(gen.parameters(), lr=GEN_LR)
    dis_optimizer = optim.Adam(dis.parameters(), lr=DIS_LR)

    # Load pretrained models
    gen.load_state_dict(torch.load(pretrained_gen, map_location=DEVICE))
    dis.load_state_dict(torch.load(pretrained_dis, map_location=DEVICE))

    dis_losses, dis_accs, pg_losses = [], [], []

    print('Starting Adversarial Learning...')

    for epoch in range(ADV_TRAIN_EPOCHS):

        print('\nEpoch %d' % (epoch + 1))

        pg_loss = MyUtils.train_generator_PG(gen, gen_optimizer, dis,
                                             NUM_PG_BATCHES, DATASET_MAX_SIZE,
                                             DEVICE)
        pg_losses.append(pg_loss)
        MyUtils.plot_png([pg_losses], ['adv_pg_losses'],
                         logs_dir + 'adv_pg_losses.png')

        adv_dis_l, adv_dis_a = MyUtils.train_discriminator(
            dis, dis_optimizer, real_samples, gen, ADV_D_STEPS, ADV_D_EPOCHS,
            DATASET_MAX_SIZE, DEVICE)
        dis_losses += adv_dis_l
        dis_accs += adv_dis_a
        MyUtils.plot_png([dis_losses], ['dis_losses'],
                         logs_dir + 'dis_losses.png')
        MyUtils.plot_png([dis_accs], ['dis_accs'], logs_dir + 'dis_accs.png')

        if epoch % 100 == 99:
            torch.save(
                gen.state_dict(),
                models_dir + run_dir + '{}_'.format(epoch + 1) + 'gen.pth')
            torch.save(
                dis.state_dict(),
                models_dir + run_dir + '{}_'.format(epoch + 1) + 'dis.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Adversarial Learning of MPOGAN')
    parser.add_argument('--run_name', type=str, help='Run name')
    parser.add_argument('--device',
                        type=str,
                        default="cuda:1",
                        help='Run device')
    parser.add_argument('--epoch',
                        type=int,
                        default=1000,
                        help='Epoch of adversarial learning')
    parser.add_argument('--gen_lr',
                        type=float,
                        default=0.00005,
                        help='LR of generator')
    parser.add_argument('--dis_lr',
                        type=float,
                        default=0.00005,
                        help='LR of discriminator')
    parser.add_argument('--adv_d_steps',
                        type=int,
                        default=1,
                        help='Adversarial training steps of discriminator')
    parser.add_argument('--adv_d_epochs',
                        type=int,
                        default=5,
                        help='Adversarial training epochs of discriminator')
    args = parser.parse_args()

    pretrained_gen = 'models/gen_models/generator_pretraining/pre_gen.pth'
    pretrained_dis = 'models/AMP_classifier/best_model/best_model.pth'

    dataset_file = 'data/positive.fasta'

    dataset_seq = MyUtils.get_fasta_dataset(dataset_file)
    print(f'Dataset size: {len(dataset_seq)}')

    torch.manual_seed(2024)
    DEVICE = args.device
    DATASET_MAX_SIZE = len(dataset_seq)
    ADV_TRAIN_EPOCHS = args.epoch
    GEN_LR = args.gen_lr
    DIS_LR = args.dis_lr
    ADV_D_STEPS = args.adv_d_steps
    ADV_D_EPOCHS = args.adv_d_epochs

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    adversarial_learning(dataset_seq, args.run_name, pretrained_gen,
                         pretrained_dis)
