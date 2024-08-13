import torch

CHARACTER_DICT = {
    'A': 1,
    'C': 2,
    'E': 3,
    'D': 4,
    'F': 5,
    'I': 6,
    'H': 7,
    'K': 8,
    'M': 9,
    'L': 10,
    'N': 11,
    'Q': 12,
    'P': 13,
    'S': 14,
    'R': 15,
    'T': 16,
    'W': 17,
    'V': 18,
    'Y': 19,
    'G': 20,
}

INDEX_DICT = {
    1: 'A',
    2: 'C',
    3: 'E',
    4: 'D',
    5: 'F',
    6: 'I',
    7: 'H',
    8: 'K',
    9: 'M',
    10: 'L',
    11: 'N',
    12: 'Q',
    13: 'P',
    14: 'S',
    15: 'R',
    16: 'T',
    17: 'W',
    18: 'V',
    19: 'Y',
    20: 'G',
}

CUDA = torch.cuda.is_available()

SEQ_LEN = (0, 25)
MAX_SEQ_LEN = SEQ_LEN[1]
VOCAB_SIZE = 22
START_LETTER = 0
BATCH_SIZE = 256
NUM_PG_BATCHES = 1

GEN_EMBEDDING_DIM = 3
GEN_HIDDEN_DIM = 128
DIS_EMBEDDING_DIM = 3
DIS_HIDDEN_DIM = 128