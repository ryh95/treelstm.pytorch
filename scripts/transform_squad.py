import copy
import json
import random

import re
# ssplit sentence in squad
from pycorenlp import StanfordCoreNLP


nlp = StanfordCoreNLP('http://localhost:9000')

def ssplit(context):
    output = nlp.annotate(context, properties={
        'annotators': 'ssplit',
        'outputFormat': 'json'
    })

    sen_pos = []

    for sentence in output['sentences']:
        sen_start_pos = sentence['tokens'][0]['characterOffsetBegin']
        sen_end_pos = sentence['tokens'][-1]['characterOffsetEnd']
        sen_pos.append((sen_start_pos, sen_end_pos))

    split_sentences = []

    for start, end in sen_pos:
        split_sentences.append(context[start:end])

    space_between_sen = []
    for i, (start, _) in enumerate(sen_pos[1:], 1):
        num_space = start - sen_pos[i - 1][1]
        space_between_sen.append(num_space)

    return split_sentences,space_between_sen


def transform_squad(original_json_file,output):
    with open(original_json_file,'r') as original_file:
        original_data = json.load(original_file)
        # data to be dumped
        id = 0
        data_list = []
        n_context = 0
        n_question = 0
        n_answer = 0
        n_error_ssplit = 0
        n_illed_context = 0
        n_sen = 0
        n_sen_list = []
        # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        for article_id,article in enumerate(original_data['data'],1):

            for para_id,paragraph in enumerate(article['paragraphs'],1):
                context = paragraph['context']

                sentences,spaces = ssplit(context)
                n_context += 1
                # check if has two same sentences
                if len(set(sentences)) != len(sentences):
                    print ('context has two same sentences')
                    n_illed_context += 1
                    continue

                len_sentences = [len(s) for s in sentences]


                for qa in paragraph['qas']:
                    n_question += 1
                    n_sen += len(sentences)
                    n_sen_list.append(n_sen)
                    for answer in qa['answers']:
                        n_answer += 1
                        # contain too many duplicate sentence pairs
                        # may remove them
                        text = answer['text']
                        answer_start = answer['answer_start']

                        # get sentence number and answer index of this sentence
                        pre_cum_lens, sen_pos = 0, 0
                        for sen_id, len_sentence in enumerate(len_sentences):
                            if pre_cum_lens + len_sentence >= answer_start:
                                sen_pos = sen_id
                                break
                            pre_cum_lens = pre_cum_lens + len_sentence + spaces[sen_id]

                        error_ssplit = False
                        for sen_id,sentence in enumerate(sentences):
                            if sen_id == sen_pos:
                                try:
                                    idx = sentence.index(text)
                                    sample = {"pair_ID":str(article_id)+'-'+str(para_id),
                                              "sentence_A":qa['question'],
                                              "sentence_B":sentence.replace('\n',''),
                                              "relatedness_score":'2' ,
                                              "answer": text,
                                              "is_adversarial": '1' if sen_id == len(sentences)-1 else '0'
                                    }
                                    data_list.append(json.dumps(sample))
                                except ValueError:
                                    error_ssplit = True
                                    n_error_ssplit += 1
                                    # print('Error ssplit')
                            else:
                                sample = {"pair_ID":str(article_id)+'-'+str(para_id),
                                          "sentence_A": qa['question'],
                                          "sentence_B": sentence.replace('\n',''),
                                          "relatedness_score": '1',
                                          "answer": 'Null',
                                          "is_adversarial": '1' if sen_id == len(sentences) - 1 else '0'
                                }
                                data_list.append(json.dumps(sample))
                                if error_ssplit == True:
                                    data_list = data_list[:-sen_id]
                                    # print (data_list[-1])
                                    break
        print ('num of context: {}'.format(n_context))
        print ('num of question: {}'.format(n_question))
        print ('num of answer: {}'.format(n_answer))
        print ('num of sentences: {}'.format(n_sen))
        print ('num of paras: {}'.format(len(n_sen_list)))
        print ('num of average sentences: {}'.format(n_sen/len(n_sen_list)))
        print ('num of error_ssplit: {}'.format(n_error_ssplit))
        print ('num of illed_context: {}'.format(n_illed_context))
        print ('num of data(duplicate): {}'.format(len(data_list)))
        fi_data_list = set(data_list)
        # sort according to int pair_ID
        def tmp_sort_func(x):
            a,b = json.loads(x)['pair_ID'].split('-')
            return int(a),int(b)
        fi_data_list = sorted(fi_data_list,key=lambda x: tmp_sort_func(x))
        print ('num of data: {}'.format(len(fi_data_list)))
        header_list = ['pair_ID','sentence_A','sentence_B','relatedness_score','answer','is_adversarial']
        with open(output,'w') as f:
            f.write('\t'.join(header_list)+'\n')
            for idx,sample in enumerate(fi_data_list,1):
                sample = json.loads(sample)
                f.write('\t'.join(sample[header] for header in header_list)+'\n')

def remove_normal_context(adversarial_normal_json_file,adversarial_json_file):
    data_list = []
    output_data_json = {"version":"1.1","data":data_list}
    with open(adversarial_normal_json_file,'r') as f_in_normal_adver:
            f_in_normal_adver_data = json.load(f_in_normal_adver)
            for article in f_in_normal_adver_data['data']:
                title = article['title']
                output_paras = []
                for para in article['paragraphs']:
                    context = para['context']
                    output_qas = []
                    for qa in para['qas']:
                        question = qa['question']
                        idx = qa['id']
                        answers = qa['answers']
                        if len(idx.split('-')) > 1:
                            # add to output qas
                            output_qas.append(qa)
                    output_para = {'context':context,'qas':output_qas}
                    # remove normal sample
                    if len(output_qas) > 0: output_paras.append(output_para)
                output_article = {'paragraphs':output_paras,'title':title}
                data_list.append(output_article)
    json.dump(output_data_json, open(adversarial_json_file,'w'))

def check_questions(check_file):
    with open(check_file,'r') as f:
        add_sen_dict = json.load(f)
        num_item_qas = []
        for item in add_sen_dict['data']:
            for paragraph in item['paragraphs']:
                num_item_qas.append(len(paragraph['qas']))

        print(sum(num_item_qas))
