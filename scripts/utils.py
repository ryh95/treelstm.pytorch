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


def filter_sent_len(file,file_out,threshold):
    new_id = 1
    with open(file,'r') as f,\
         open(file_out,'w') as file_out:
        file_out.write(
            'pair_ID' + '\t' + 'sentence_A' + '\t' + 'sentence_B' + '\t' + 'relatedness_score' + '\t' + 'entailment_judgment' + '\n')
        f.readline()
        for line in f:
            i, a, b, sim, ent = line.strip().split('\t')
            if len(b) > 100:
                continue
            file_out.write('\t'.join([str(new_id),a,b,sim,ent])+'\n')
            new_id += 1
            if new_id > threshold:
                break

if __name__ == '__main__':
    filter_sent_len('SICK_squad_trial.txt','SICK_squad_dev.txt',4000)
    analysis_label('SICK_squad_dev.txt')
    count_sent_len('SICK_squad_dev.txt')