'''Discriminator (i.e., the antimicrobial activity preidctor (LLM-AAP)) pretraining'''
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models import Discriminator
import sys
import os
import pandas as pd
import MyUtils

DEVICE = 'cuda:0'
MAX_SEQ_LEN = 25
LR = 0.00005
DROPOUT = 0.1
EPOCH = 200
BATCH_SIZE = 256
np.random.seed(2024)


class FastaDataset(Dataset):

    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def prepare_data(pos_file, neg_file):
    '''Prepare the dataset for training the discriminator'''

    pos_seqs = MyUtils.get_fasta_dataset(pos_file)
    neg_seqs = MyUtils.get_fasta_dataset(neg_file)

    print('Positive sequences: ', len(pos_seqs))
    print('Negative sequences: ', len(neg_seqs))

    # Random shuffle the sequences
    np.random.shuffle(pos_seqs)
    np.random.shuffle(neg_seqs)

    # Split the dataset into train, validation and test set (8:1:1)
    pos_train = pos_seqs[:int(len(pos_seqs) * 0.8)]
    pos_val = pos_seqs[int(len(pos_seqs) * 0.8):int(len(pos_seqs) * 0.9)]
    pos_test = pos_seqs[int(len(pos_seqs) * 0.9):]

    neg_train = neg_seqs[:int(len(neg_seqs) * 0.8)]
    neg_val = neg_seqs[int(len(neg_seqs) * 0.8):int(len(neg_seqs) * 0.9)]
    neg_test = neg_seqs[int(len(neg_seqs) * 0.9):]

    # Construct the training, validation and test set
    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    # Construct the labels
    train_labels = [1] * len(pos_train) + [0] * len(neg_train)
    val_labels = [1] * len(pos_val) + [0] * len(neg_val)
    test_labels = [1] * len(pos_test) + [0] * len(neg_test)

    return (train, val, test, train_labels, val_labels, test_labels)


def train_model(data: tuple,
                lr=LR,
                dropout=DROPOUT,
                model_path='./models/AMP_classifier/default/'):
    '''Train the discriminator model'''

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train, val, test, train_labels, val_labels, test_labels = data

    train_dataset = FastaDataset(train, train_labels)
    val_dataset = FastaDataset(val, val_labels)
    test_dataset = FastaDataset(test, test_labels)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define the model
    model = Discriminator(DEVICE, MAX_SEQ_LEN + 2, dropout)
    model.to(DEVICE)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define the loss function
    criterion = torch.nn.BCELoss()

    # Train the model
    train_loss, train_epochs_loss, train_epochs_acc = [], [], []
    val_loss, val_epochs_loss, val_epochs_acc = [], [], []

    for epoch in range(EPOCH):

        print('Epoch: %d' % epoch)
        model.train()
        train_epoch_loss = []
        train_epoch_acc = 0

        for i, batch in enumerate(train_loader):
            seqs, labels = batch
            labels = torch.nn.functional.one_hot(
                torch.tensor(labels).to(DEVICE), 2).float()

            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_epoch_loss.append(loss.item())

            train_argmax = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            for j in range(len(labels)):
                if train_argmax[j] == labels.cpu().detach().numpy()[j][1]:
                    train_epoch_acc += 1

            # myUtils.plot_png([train_loss], ['train loss'],
            #                      './models/AMP_classifier/train_loss.png')

            # if i % 20 == 0:
            #     print('Epoch: %d, Batch: %d, Loss: %.4f' % (epoch, i, loss.item()))

        train_acc = train_epoch_acc / len(train_labels)
        train_epochs_acc.append(train_acc)
        train_epochs_loss.append(np.mean(train_epoch_loss))

        # Evaluate the model on the validation set
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            val_epoch_acc = 0
            for i, batch in enumerate(val_loader):
                seqs, labels = batch
                labels = torch.nn.functional.one_hot(
                    torch.tensor(labels).to(DEVICE), 2).float()

                outputs = model(seqs)
                loss = criterion(outputs, labels)

                val_loss.append(loss.item())
                val_epoch_loss.append(loss.item())

                val_argmax = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                for j in range(len(labels)):
                    if val_argmax[j] == labels.cpu().detach().numpy()[j][1]:
                        val_epoch_acc += 1

                # myUtils.plot_png([val_loss], ['val loss'],
                #                      './models/AMP_classifier/val_loss.png')

            val_acc = val_epoch_acc / len(val_labels)
            val_epochs_acc.append(val_acc)
            val_epochs_loss.append(np.mean(val_epoch_loss))

        print(
            '\tTrain Loss:\t%.4f\tTrain Acc:\t%.4f\n\tVal Loss:\t%.4f\tVal Acc:\t%.4f\n'
            % (np.mean(train_epoch_loss), train_acc, np.mean(val_epoch_loss),
               val_acc))

        MyUtils.plot_png([train_epochs_loss, val_epochs_loss],
                         ['train epoch loss', 'val epoch loss'],
                         model_path + 'loss.png',
                         is_smooth=False)
        MyUtils.plot_png([train_epochs_acc, val_epochs_acc],
                         ['train epoch acc', 'val epoch acc'],
                         model_path + 'acc.png',
                         is_smooth=False)

        # Save the best model
        if val_acc == max(val_epochs_acc):
            torch.save(model.state_dict(), model_path + 'best_model.pth')

        sys.stdout.flush()

    # Evaluate the model on the test set
    with torch.no_grad():
        model.eval()
        test_epoch_acc = 0
        for i, batch in enumerate(test_loader):
            seqs, labels = batch
            labels = torch.nn.functional.one_hot(
                torch.tensor(labels).to(DEVICE), 2).float()

            outputs = model(seqs)
            # loss = criterion(outputs, labels)

            test_argmax = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            for j in range(len(labels)):
                if test_argmax[j] == labels.cpu().detach().numpy()[j][1]:
                    test_epoch_acc += 1

        test_acc = test_epoch_acc / len(test_labels)
        print(f'\tTest Acc:\t{test_acc}')

    return train_epochs_loss, train_epochs_acc, val_epochs_loss, val_epochs_acc, test_acc


def grid_search():

    data = load_train_val_test_data(
        train_file='./data/AMP_classifier_data/train.csv',
        val_file='./data/AMP_classifier_data/val.csv',
        test_file='./data/AMP_classifier_data/test.csv')

    lr_list = [0.0005, 0.0001, 0.00005]
    dropout_list = [0.1]
    best_acc = 0
    best_lr = 0
    best_dropout = 0
    best_test_acc = 0

    for lr in lr_list:
        for dropout in dropout_list:
            model_path = f'./models/AMP_classifier/lr_{lr}_dropout_{dropout}/'
            train_epochs_loss, train_epochs_acc, val_epochs_loss, val_epochs_acc, test_acc = train_model(
                data, lr, dropout, model_path=model_path)

            df = pd.DataFrame({
                'epoch': range(EPOCH),
                'train_epochs_loss': train_epochs_loss,
                'train_epochs_acc': train_epochs_acc,
                'val_epochs_loss': val_epochs_loss,
                'val_epochs_acc': val_epochs_acc
            })
            df.to_csv(model_path + 'result.csv')

            with open(model_path + 'test_acc.txt', 'w') as f:
                f.write(str(test_acc))

            # Log the best hyperparameters
            if max(val_epochs_acc) > best_acc:
                best_acc = max(val_epochs_acc)
                best_lr = lr
                best_dropout = dropout
                best_test_acc = test_acc

    print(
        f'Best lr: {best_lr}\tBest dropout: {best_dropout}\tBest acc: {best_acc}\tBest test acc: {best_test_acc}'
    )


def construct_train_val_test_data(pos_file, neg_file, train_file, val_file,
                             test_file):
    data = prepare_data(pos_file, neg_file)
    train, val, test, train_labels, val_labels, test_labels = data

    train_df = pd.DataFrame({'sequence': train, 'amp': train_labels})
    train_df.to_csv(train_file, index=False)

    val_df = pd.DataFrame({'sequence': val, 'amp': val_labels})
    val_df.to_csv(val_file, index=False)

    test_df = pd.DataFrame({'sequence': test, 'amp': test_labels})
    test_df.to_csv(test_file, index=False)


def load_train_val_test_data(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file, keep_default_na=False)
    val_df = pd.read_csv(val_file, keep_default_na=False)
    test_df = pd.read_csv(test_file, keep_default_na=False)

    train_seqs = list(train_df['sequence'])
    train_labels = list(train_df['amp'])
    val_seqs = list(val_df['sequence'])
    val_labels = list(val_df['amp'])
    test_seqs = list(test_df['sequence'])
    test_labels = list(test_df['amp'])

    return (train_seqs, val_seqs, test_seqs, train_labels, val_labels,
            test_labels)


if __name__ == '__main__':
    # save_train_val_test_data(pos_file='./data/MPOGAN_dataset/positive.fasta',
    #                          neg_file='./data/MPOGAN_dataset/negative.fasta',
    #                          train_file='./data/AMP_classifier_data/train.csv',
    #                          val_file='./data/AMP_classifier_data/val.csv',
    #                          test_file='./data/AMP_classifier_data/test.csv')

    # grid_search()

    train_model(
        load_train_val_test_data(
            train_file='./data/AMP_classifier_data/train.csv',
            val_file='./data/AMP_classifier_data/val.csv',
            test_file='./data/AMP_classifier_data/test.csv'),
        lr=0.00005,
        dropout=0.1,
        model_path='./models/AMP_classifier/best_model/')
