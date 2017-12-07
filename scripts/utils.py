# -*- coding: utf-8 -*-
import random
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
    txt_in_file is some output file after transform-squad processed
    like SICK_squad_test_add_one_sent_adver.txt
    SICK_squad_test_add_any_adver.txt

    since questions may have multiple answers
    so may have following sentence pairs
    e.g.
    1-3-0	What was the first Super Bowl to use the standardized logo template?
    On June 4, 2014, the NFL announced that the practice of branding Super Bowl games with Roman numerals, a practice established at Super Bowl V, would be temporarily suspended, and that the game would be named using Arabic numerals as Super Bowl 50 as opposed to Super Bowl L.
    1	Null	0
    1-3-0	What was the first Super Bowl to use the standardized logo template?
    On June 4, 2014, the NFL announced that the practice of branding Super Bowl games with Roman numerals, a practice established at Super Bowl V, would be temporarily suspended, and that the game would be named using Arabic numerals as Super Bowl 50 as opposed to Super Bowl L.
    2	V	0
    one of this answer is Super Bowl XLV
    so the first pair result is null
    but another answer is V
    so the second pair result is V

    but the sentence is actually can answer the question
    so we should remove the first pair
    this function is for this purpose
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
            a,b,c = x.split('\t')[0].split('-')
            return int(a),int(b),int(c)
        out_list = sorted(set(out_list),key=lambda x: tmp_sort_func(x))
        for data in out_list:
            f_out.write(data)

def refilter_sen_pair(txt_in_file,txt_out_file):
    '''
    txt_in_file is some output file after filter_sen_pair processed
    e.g.
    SICK_squad_test_add_any_adver_filter.txt
    ..

    although filter_sen_pair removed some of sentence pair, it still has problems
    e.g.
    1-2-0	Where hotel did the Panthers stay at?
    The Panthers used the San Jose State practice facility and stayed at the San Jose Marriott.
    2	the San Jose Marriott	0
    1-2-0	Where hotel did the Panthers stay at?
    The Panthers used the San Jose State practice facility and stayed at the San Jose Marriott.
    2	San Jose Marriott.	0

    both sentence pair has same answer sentence because multiple answer spans are in this sentence
    but in order to select answer sentence one would be fine
    two or more may be too esay for model to select it out

    (maybe has no effect? if a model knows a sentence is answer sentence and suppose there are 3 duplicates
    those would all in the forward position
    if a model can't figure out answer sentence then those would all in the last position
    so when use max to find answer sentence both would have no effect
    this needs to verify)

    this could combine into filter_sen_pair
    :param txt_in_file:
    :param txt_out_file:
    :return:
    '''
    def filter_v_list(v_list):
        # only 1 in 2 will filter 1 in value_list
        filter_v = False
        sim_list = [int(v.split('\t')[0]) for v in v_list]
        if sum(sim_list) == 2*len(sim_list):
            filter_v = True

        if filter_v:
            return [random.choice(v_list)]
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
            a,b,c = x.split('\t')[0].split('-')
            return int(a),int(b),int(c)
        out_list = sorted(set(out_list),key=lambda x: tmp_sort_func(x))
        for data in out_list:
            f_out.write(data)

def check_context(txt_in_file):
    '''

    :param txt_in_file: test adversarial refilter file
    :return:
    '''
    context_id_set = set()
    with open(txt_in_file,'r') as f_in:
        f_in.readline()
        for line in f_in:
            pair_id = line.split('\t')[0]
            context_id_set.add('-'.join(pair_id.split('-')[:-1]))
    print(len(context_id_set))

def merge_sen_pair_result_to_context_pair(txt_in_file,txt_out_file):
    '''
    e.g.
    SICK_squad_test_add_any_adver_refilter_result.txt
    :param txt_in_file:
    :return:
    '''
    with open(txt_in_file,'r') as f_in,\
         open(txt_out_file,'w') as f_out:
        f_in.readline()
        context_pair_dict = defaultdict(list)
        for line in f_in:
            line = line.strip('\n')
            pair_ID,sen_A,*rest = line.split('\t')
            context_pair_ID = '-'.join(pair_ID.split('-')[:-1])
            context_pair_dict[context_pair_ID+'\t'+sen_A].append('\t'.join(rest))

        # sort dict by keys
        def tmp_sort_func(x):
            pair_ID = x.split('\t')[0]
            a,b = pair_ID.split('-')
            return int(a),int(b)
        new_keys = sorted(context_pair_dict.keys(),key=lambda x: tmp_sort_func(x))

        for key in new_keys:
            value_list = context_pair_dict[key]
            value = max(value_list,key=lambda x: float(x.split('\t')[1]))
            f_out.write(key+'\t'+value+'\n')

def evaluate_context_pair(txt_in_file):
    with open(txt_in_file,'r') as f_in:
        n_correct = 0
        n_all = 0
        for line in f_in:
            line = line.strip('\n')
            context_pair_ID,sen_A,sen_B,pred_score,re_score,is_adver = line.split('\t')
            # since all the choosen sen_B is the correct sentences that model thinks
            # so just calculate how many re_score is 2 would be precision
            if re_score == '2':
                n_correct += 1
            n_all += 1
        print(n_correct,n_all)
        print(n_correct/n_all)

def merge_context_result_sen_pair(context_result_file,S_Q_a_pair_file,merged_file):
    '''
    after merge_sen_pair_result_to_context_pair
    add answer for context_result
    i.e.
    produce <S,Q,a> pair
    :param context_result_file:
    :param S_Q_a_pair_file:
    :param merged_file:
    :return:
    '''
    # get answer dict from S_Q_a_pair_file
    S_Q_a_pair_dict = defaultdict(list)
    with open(S_Q_a_pair_file,'r') as f:
        # remove header
        f.readline()
        for line in f:
            line = line.strip('\n')
            idx,sen_A,sen_B,re_score,answer,is_adv = line.split('\t')
            if re_score == '2':
                S_Q_a_pair_dict[sen_A].append(answer)

    with open(context_result_file,'r') as f,\
         open(merged_file,'w') as f_out:
        for line in f:
            line = line.strip('\n')
            idx,sen_A,sen_B,pred_score,re_score,is_adv = line.split('\t')
            answer_list = S_Q_a_pair_dict[sen_A]
            for answer in answer_list:
                f_out.write(sen_A+'\t'+sen_B+'\t'+answer+'\t'+re_score+'\n')

def filter_wrong_context(txt_merged_file,filtered_merge_file):
    with open(txt_merged_file,'r') as f,\
         open(filtered_merge_file,'w') as f_out:
        for line in f:
            line = line.strip('\n')
            # test = line.split('\t')
            sen_A,sen_B,ans,re_score = line.split('\t')
            if (ans in sen_B) and re_score == '2':
                f_out.write(sen_A+'\t'+sen_B+'\t'+ans+'\n')