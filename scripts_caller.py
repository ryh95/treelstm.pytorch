from scripts.transform_squad import transform_squad

# transform_squad
print('=' * 80)
# print('Transforming squad training')
print('=' * 80)
transform_squad('adversarial_data_adver.json','SICK_squad_test_add_any_debug.txt')
print('=' * 80)
# print('Transforming squad dev')
print('=' * 80)
# remove_normal_context('sample1k-HCVerifySample.json', 'SICK_squad_test_add_one_sent_adver.json')
# check_questions('SICK_squad_test_add_one_sent_adver.json')
# transform_squad('SICK_squad_test_add_one_sent_adver.json','SICK_squad_test_add_one_sent_adver.txt')

# analysis_label('SICK_squad_test_add_one_sent_adver.txt')
# print('=' * 80)
# analysis_label('SICK_squad_trial.txt')

# scripts/utils
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
# filter_sen_pair('SICK_squad_test_add_any.txt','SICK_squad_test_add_any_adver_filter.txt')
# refilter_sen_pair('SICK_squad_test_add_any_adver_filter.txt',
#                   'SICK_squad_test_add_any_adver_refilter.txt')

# check_context('SICK_squad_test_add_any_adver_refilter.txt')
# print (check_untokenizable())

# f1 = 'SICK_squad_train_all.txt'
# f2 = 'SICK_squad_train.txt'
# analysis_label(f1)
# analysis_label(f2)