# -*- coding: utf-8 -*-
import json
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
def analysis_label(sick_file):
    n_pos = 0
    n_neg = 0
    n = 0
    with open(sick_file,'r') as f:
        f.readline()
        for line in f:
            i, a, b, sim, ent,*_ = line.strip().split('\t')
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
    print (float(n_token_a)/n_line,float(n_token_b)/n_line)


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

def filter_sen_pair(txt_in_file,txt_out_file):
    '''
    txt_file is some file like SICK_squad_test_add_one_sent_adver.txt
    :param txt_file:
    :return: filtered txt file
    '''
    def filter_v_list(v_list):
        # only 1 in 2 will filter 1 in value_list
        filter_v = False
        sim_list = [v.split('\t')[0] for v in v_list]
        if '1' in sim_list and '2' in sim_list:
            filter_v = True

        if filter_v:
            for v in v_list:
                if v.split('\t')[0] == '1':
                    v_list.remove(v)
        return v_list

    with open(txt_in_file,'r') as f_in,\
         open(txt_out_file,'w') as f_out:
        f_out.write(f_in.readline())
        data_dict = defaultdict(list)
        out_list = []
        for line in f_in:
            *key,sim,ans,is_adv = line.split('\t')
            str_key = '\t'.join(key)
            str_value = '\t'.join([sim,ans,is_adv])
            data_dict[str_key].append(str_value)

        # filter
        for key in data_dict:
            v_list = filter_v_list(data_dict[key])
            for v in v_list:
                out_list.append(key+'\t'+v)
        def tmp_sort_func(x):
            a,b = x.split('\t')[0].split('-')
            return int(a),int(b)
        out_list = sorted(set(out_list),key=lambda x: tmp_sort_func(x))
        for data in out_list:
            f_out.write(data)

def merge_add_any_json(merged_adver_json_file,*file_in_list):
    data_list = []
    output_data_json = {"version": "1.1", "data": data_list}
    for file in file_in_list:
        with open(file, 'r') as f_in_adver:
            f_in_adver_data = json.load(f_in_adver)
            adver_data_list = f_in_adver_data['data']
            data_list += adver_data_list

    json.dump(output_data_json, open(merged_adver_json_file, 'w'))

def split_sentence_answer_span(txt_in_file,txt_out_file,num_samples):
    '''
    txt_file is some file like SICK_squad_test_add_one_sent_adver.txt
    :param txt_in_file:
    :return:
    '''
    with open(txt_in_file,'r') as f,\
         open(txt_out_file,'w') as f_out:
        f.readline()
        sample_num = 0
        for line in f:
            i, a, b, sim, ans,*_ = line.strip().split('\t')
            if sim == '2':
                f_out.write(a+'\t'+b+'\t'+ans+'\n')
                sample_num += 1
            if sample_num >= num_samples:
                break
