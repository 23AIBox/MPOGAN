##############################################################################
#ToxinPred2.0 is developed for predicting toxin and non toxin      #
#protein from their primary sequence. It is developed by Prof G. P. S.       #
#Raghava's group. Please cite : ToxinPred 2.0                                  #
# ############################################################################
import warnings
import os
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')


def aac_comp(seqs, out):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    with open(out, 'w') as f:
        for j in seqs:
            composition = {}
            seq_length = len(j)

            for i in std:
                count = j.count(i)
                composition[i] = (count / seq_length) * 100

            aac_values = [composition[i] for i in std]
            aac_str = ", ".join(["%.2f" % val for val in aac_values])
            f.write(aac_str + '\n')


def prediction(inputfile, model):
    df = pd.DataFrame()
    clf = joblib.load(model)
    data_test = np.loadtxt(inputfile, delimiter=',')
    y_p_score1 = clf.predict_proba(data_test)
    y_p_s1 = y_p_score1.tolist()
    df = pd.DataFrame(y_p_s1)
    df_1 = df.iloc[:, -1]
    return df_1.to_numpy()


def toxinpred2(seqs, tmp_dir):
    aac_comp(seqs, tmp_dir + 'seq.aac')
    os.system(f"perl -pi -e 's/,$//g' {tmp_dir}seq.aac")
    pred = prediction(tmp_dir + 'seq.aac', './models/toxinpred2/RF_model')
    os.remove(tmp_dir + 'seq.aac')
    return pred
