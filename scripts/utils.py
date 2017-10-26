# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
def analysis_label(sick_file):
    n_pos = 0
    n_neg = 0
    n = 0
    with open(sick_file,'r') as f:
        f.readline()
        for line in f:
            i, a, b, sim, ent = line.strip().split('\t')
            if int(sim) == 2:
                n_pos += 1
            elif int(sim) == 1:
                n_neg += 1
            n += 1
    print ('num of positive samples: {}'.format(n_pos))
    print ('num of negetive samples: {}'.format(n_neg))
    print ('num of samples: {}'.format(n))

def count_sent_len(file):

    n_line = 0
    n_token_a = 0
    n_token_b = 0
    with open(file,'r') as f:
        f.readline()
        for line in f:
            i, a, b, sim, ent = line.strip().split('\t')
            n_token_a += len(a)
            n_token_b += len(b)
            n_line += 1
    print float(n_token_a)/n_line,float(n_token_b)/n_line


def filter_sent_len(file,file_out,n_sample_threshold,len_a_threshold=10000,len_b_threshold=10000,ratio=0.5):
    new_id = 1
    n_pos = 0
    n_neg = 0
    n_pos_th = int(ratio * n_sample_threshold)
    n_neg_th = n_sample_threshold - n_pos_th

    with open(file,'r') as f,\
         open(file_out,'w') as file_out:
        file_out.write(
            'pair_ID' + '\t' + 'sentence_A' + '\t' + 'sentence_B' + '\t' + 'relatedness_score' + '\t' + 'entailment_judgment' + '\n')
        f.readline()
        for line in f:
            i, a, b, sim, ent = line.strip().split('\t')
            if len(b) <= len_b_threshold and len(a) <= len_a_threshold:
                if int(sim) == 2 and n_pos < n_pos_th:
                    # write
                    file_out.write('\t'.join([str(new_id), a, b, sim, ent]) + '\n')
                    n_pos += 1
                    new_id += 1
                if int(sim) == 1 and n_neg < n_neg_th:
                    # write
                    file_out.write('\t'.join([str(new_id), a, b, sim, ent]) + '\n')
                    n_neg += 1
                    new_id += 1

            if new_id > n_sample_threshold:
                break

def displot_sick(file):
    sims = []
    with open(file,'r') as f:
        f.readline()
        for line in f:
            i, a, b, sim, ent = line.strip().split('\t')
            sims.append(float(sim))
    sns.distplot(sims)
    plt.show()

# Todo: finish check_untokenizable
# def check_untokenizable():
#     a = '₹'.decode('utf8')
#     return a.replace(u'₹','h')

if __name__ == '__main__':
    n_sample_th = 3000
    f_in = 'SICK_squad_dev_all.txt'
    f_out = 'SICK_squad_dev_'+ str(n_sample_th) + '.txt'
    len_b_th = 100

    filter_sent_len(f_in,f_out,n_sample_th,len_b_threshold=len_b_th)
    analysis_label(f_out)
    count_sent_len(f_out)
    displot_sick(f_out)

    # print (check_untokenizable())

    # f1 = 'SICK_squad_train_all.txt'
    # f2 = 'SICK_squad_train.txt'
    # analysis_label(f1)
    # analysis_label(f2)