# MPOGAN: A Multi-Property Optimizing Generative Adversarial Network for de novo Antimicrobial Peptides Design

## Overview of MPOGAN

This study introduces a Multi-Property Optimizing Generative Adversarial Network (MPOGAN). We propose an extended feedback-loop GAN framework, specifically designed to tackle the scarcity of high-quality AMPs with multiple desired properties as training data. By learning from iteratively generated data, MPOGAN can design novel AMPs with potent antimicrobial activity, reduced cytotoxicity, and diversity.
The framework of MPOGAN includes a pre-training stage and a multi-property optimizing (MPO) stage. The pre-training stage enables MPOGAN to learn the general characteristics of AMPs. 
Subsequently, the MPO stage is proposed to optimize multiple desired properties iteratively. 

## Set up environment

Setup the required environment using requirements.txt with python. While in the project directory run:

```bash
pip install -r requirement.txt
```

## Train the model

1. Generator pretraining

```bash
python GeneratorPretraining.py --run_name 'generator_pretraining'
```

2. Discriminator pretraining

```bash
python DiscriminatorPretraining.py
```

3. Adversarial learning

```bash
python AdversarialLearning.py --run_name 'adversarial_learning' --device 'cuda:0' --epoch 1000 --dis_lr 0.00005 --adv_d_steps 1 --adv_d_epochs 5
```

4. Multi-property optimizing

```bash
python MultiPropertyOptim.py --run_name 'MPOGAN' --epoch 1000 --device 'cuda:0' --dataset_max_size 1000 --epoch 700 --gen_lr 0.00005 --dis_lr 0.00005 --ur 0.25 --ugr 0.7 --with_toxin 1 --with_cdhit 1 --with_stable_update 1 --amp_threshold 0.8 --toxin_threshold 0.7 --cdhit_threshold 0.6
```

## Generate AMP candidates

```bash
python generateCandidates.py --model_id 700 --num_outputs 50000 --run_name MPOGAN_finetuning
```

## Reference

If you find MPOGAN useful, please consider citing: 

```
@article{liu2024multi,
  title={A Multi-Property Optimizing Generative Adversarial Network for de novo Antimicrobial Peptide Design},
  author={Liu, Jiaming and Cui, Tao and Wang, Tao and Zeng, Xi and Niu, Yinbo and Jiao, Shaoqing and Lu, Dazhi and Wang, Jun and Xiao, Shuyuan and Xie, Dongna and others},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
