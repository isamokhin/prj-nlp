from tagger import PerceptronTagger

import conllu
import gzip

fname = '../uk_iu-ud-train.conllu.gz'
with gzip.open(fname, 'rb') as f:
    raw_train = f.read().decode()
    
train_set = conllu.parse(raw_train)

train_corpus = []
for sent in train_set:
    train_corpus.append([(w['form'], w['upostag']) for w in sent])
    
def perceptron_train_and_save(train_corpus, nr_iter=10, 
                              fname='uk_perceptron_tagger.pickle'):
    perc_train = []
    for sent in train_corpus:
        words = [w[0] for w in sent]
        tags = [w[1] for w in sent]
        perc_train.append((words, tags))
    p = PerceptronTagger(load=False)
    p.train(perc_train, nr_iter=nr_iter, save_loc=fname)
    
print('Training and saving the perceptron POS tagger.')
perceptron_train_and_save(train_corpus, 20)
print('Done!')
