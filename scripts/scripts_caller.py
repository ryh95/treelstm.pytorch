
# scripts/utils
from scripts.utils import split_sentence_answer_span
from scripts.transform_squad import transform_squad
n_sample_th = 90000
# train/dev
file_type = 'train'
f_in = 'SICK_squad_'+file_type+'_all.txt'
f_out = 'SICK_squad_'+file_type+'_'+ str(n_sample_th) + '.txt'
len_b_th = 1000

# filter_sent_len(f_in,f_out,n_sample_th,len_b_threshold=len_b_th)
# analysis_label(f_out)
# count_sent_len(f_out)
# displot_sick(f_out)
# filter_sen_pair('SICK_squad_dev.txt','SICK_squad_dev_filter.txt')
# file_in_list = ['adversarial_data_'+str(i)+'.json' for i in range(10)]
# merge_add_any_json('adversarial_data.json',*file_in_list)

split_sentence_answer_span('SICK_squad_dev_filter.txt','dev.txt',1000)

# print (check_untokenizable())

# f1 = 'SICK_squad_train_all.txt'
# f2 = 'SICK_squad_train.txt'
# analysis_label(f1)
# analysis_label(f2)

# scripts/transform-squad
print('=' * 80)
# print('Transforming squad training')
print('=' * 80)
# transform_squad('sample1k-HCVerifyAll.json','SICK_squad_test_add_sent.txt')
print('=' * 80)
# print('Transforming squad dev')
print('=' * 80)
# remove_normal_context('sample1k-HCVerifySample.json', 'SICK_squad_test_add_one_sent_adver.json')
# check_questions('SICK_squad_test_add_one_sent_adver.json')
transform_squad('dev-v1.1.json','SICK_squad_dev.txt')

# analysis_label('SICK_squad_test_add_one_sent_adver.txt')
# print('=' * 80)
# analysis_label('SICK_squad_trial.txt')