import copy
import json
import random
from nltk.tokenize import WordPunctTokenizer

import nltk
# ssplit sentence in squad
from my_tools.core_nlp import StanfordCoreNLP

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

def analysis_label(sick_file):
    n_pos = 0
    n_neg = 0
    n = 0
    with open(sick_file,'r') as f:
        f.readline()
        for line in f:
            i, a, b, sim, ent = line.strip().split('\t')
            if int(sim) == 1:
                n_pos += 1
            elif int(sim) == 0:
                n_neg += 1
            n = i
    print ('num of positive samples: {}'.format(n_pos))
    print ('num of negetive samples: {}'.format(n_neg))
    print ('num of samples: {}'.format(n))

def transform_squad(original_json_file,output):
    with open(original_json_file,'r') as original_file:
        original_data = json.load(original_file)
        # data to be dumped
        id = 0
        data_list = []
        n_question = 0
        n_answer = 0
        n_error_ssplit = 0
        # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        for article in original_data['data']:

            for para_id,paragraph in enumerate(article['paragraphs']):
                context = paragraph['context']

                sentences,spaces = ssplit(context)
                # check if has two same sentences
                assert len(set(sentences)) == len(sentences),'context has two same sentences'

                len_sentences = [len(s) for s in sentences]


                for qa in paragraph['qas']:
                    n_question += 1
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
                                    sample = {
                                              "sentence_A":qa['question'],
                                              "sentence_B":sentence.replace('\n',''),
                                              "relatedness_score":1 ,
                                              "entailment_judgment": 'Null'
                                    }
                                    data_list.append(json.dumps(sample))
                                except ValueError:
                                    error_ssplit = True
                                    n_error_ssplit += 1
                                    print('Error ssplit')
                            else:
                                sample = {
                                          "sentence_A": qa['question'],
                                          "sentence_B": sentence.replace('\n',''),
                                          "relatedness_score": 0,
                                          "entailment_judgment": 'Null'
                                }
                                data_list.append(json.dumps(sample))
                                if error_ssplit == True:
                                    data_list = data_list[:-sen_id]
                                    # print (data_list[-1])
                                    break

        print ('num of question: {}'.format(n_question))
        print ('num of answer: {}'.format(n_answer))
        print ('num of error_ssplit: {}'.format(n_error_ssplit))
        print ('num of data(duplicate): {}'.format(len(data_list)))
        fi_data_list = set(data_list)
        print ('num of data: {}'.format(len(fi_data_list)))
        with open(output,'w') as f:
            f.write('pair_ID'+'\t'+'sentence_A'+'\t'+'sentence_B'+'\t'+'relatedness_score'+'\t'+'entailment_judgment'+'\n')
            for idx,sample in enumerate(fi_data_list,1):
                sample = json.loads(sample)
                f.write(str(idx)+'\t'+sample['sentence_A']+'\t'+sample['sentence_B']+'\t'+
                        str(sample['relatedness_score'])+'\t'+sample['entailment_judgment']+'\n')

if __name__ == '__main__':
    analysis_label('output_2.txt')
    # transform_squad('none_n1000_k1_s0.json','output_2.txt')