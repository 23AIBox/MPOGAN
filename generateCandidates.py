import argparse
from MyUtils import generate_seqs, seqs_to_file


def main(model_id, num_outputs, run_name):
    output_file_name = f'epoch{model_id}_outputs_{num_outputs}.fasta'

    fbseqgan_seqs = generate_seqs(run_name,
                                  model_name=f'{model_id}_gen.pth',
                                  sample_nums=num_outputs)
    seqs_to_file(fbseqgan_seqs, f'./logs/{run_name}/{output_file_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate candidates using a trained MPOGAN model')
    parser.add_argument('--model_id', type=int, help='Model ID')
    parser.add_argument('--num_outputs',
                        type=int,
                        help='Number of sequences to generate')
    parser.add_argument('--run_name', type=str, help='Name of the run')
    args = parser.parse_args()

    main(args.model_id, args.num_outputs, args.run_name)
