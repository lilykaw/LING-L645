import sys, os, re, pickle, math
from collections import defaultdict, OrderedDict, Counter
from typing import OrderedDict

def preprocess(txt):
    punctuations = ".!?"
    new_txt = []
    for sent in txt:
        sent = "<BOS> " + sent    
        new_sent = ""
        for char in sent:
            if char in punctuations:
                new_sent += " " 
            new_sent += char
        new_sent = re.sub('\n', '', new_sent)
        new_txt.append(new_sent)
    return new_txt

def make_bigrams(txt):
    bigrams = []
    for line in txt:
        line = line.split(" ")
        for i,token in enumerate(line):
            if i!=0:
                bigrams.append((line[i-1], line[i]))
    return bigrams

# find perplexity of train corpus, on unigrams vs bigrams
## formula: perplexity = prob ** (-1/len(tokens))
def get_perplexity(uni_dict, bi_dict):
    uni_product_prob = 1.0
    for token in uni_dict:
        uni_product_prob *= uni_dict[token]/(sum(uni_dict.values()))
    unigram_perplex = uni_product_prob ** (-1/(sum(uni_dict.values())))

    bi_product_prob = 1.0
    for bigram in bi_dict:
        bi_product_prob *= bi_dict[bigram]
    bigram_perplex = bi_product_prob ** (-1/(sum(uni_dict.values())))

    return (unigram_perplex, bigram_perplex)



##########################################################################################
### TRAIN ################################################################################
##########################################################################################
print("Reading training data...\n")
with open('train.txt', 'r') as file:
    train = file.readlines()

# create unigrams and bigrams, plus their respective counters
print("Calculating train bigram probabilities...\n")
new_train = preprocess(train)
unigrams = [token for line in new_train for token in line.split(" ")]
uni_counter = Counter(unigrams)
print(f"Train Unigrams: {uni_counter}\n")
"""
Unigrams: Counter({
    '<BOS>': 6, 
    'you': 5, 
    '?': 5, 
    'are': 4, 
    'tired': 2, 
    'in': 2, 
    'still': 1, 
    'here': 1, 
    'where': 1, 
    'i': 1, 
    'am': 1, 
    '.': 1, 
    'england': 1, 
    'were': 1, 
    'mexico': 1
})
"""

bigrams = make_bigrams(new_train)
bi_counter = Counter(bigrams)
print(f"Train Bigrams: {bi_counter}\n")
"""
Counter({
    ('are', 'you'): 4, 
    ('<BOS>', 'are'): 3, 
    ('you', 'in'): 2, 
    ('you', 'still'): 1, 
    ('still', 'here'): 1, 
    ('here', '?'): 1, 
    ('<BOS>', 'where'): 1, 
    ('where', 'are'): 1, 
    ('you', '?'): 1, 
    ('you', 'tired'): 1, 
    ('tired', '?'): 1, 
    ('<BOS>', 'i'): 1, 
    ('i', 'am'): 1, 
    ('am', 'tired'): 1, 
    ('tired', '.'): 1, 
    ('in', 'england'): 1, 
    ('england', '?'): 1, 
    ('<BOS>', 'were'): 1, 
    ('were', 'you'): 1, 
    ('in', 'mexico'): 1, 
    ('mexico', '?'): 1
})
"""


# normalize each bigram by the frequency of the first unigram
print("Making model...\n")
model = defaultdict(lambda : defaultdict(float)) 
for bigram in bi_counter: # Calculate probabilities
    model[bigram] = bi_counter[bigram] / uni_counter[bigram[0]]
# print(f"Train Model (bigram dict): {model}\n")
"""
Model: defaultdict(<function <lambda> at 0x7f8390089550>, {
    ('<BOS>', 'are'): 0.5, 
    ('are', 'you'): 1.0, 
    ('you', 'still'): 0.2, 
    ('still', 'here'): 1.0, 
    ('here', '?'): 1.0, 
    ('<BOS>', 'where'): 0.16666666666666666, 
    ('where', 'are'): 1.0, 
    ('you', '?'): 0.2, 
    ('you', 'tired'): 0.2, 
    ('tired', '?'): 0.5, 
    ('<BOS>', 'i'): 0.16666666666666666, 
    ('i', 'am'): 1.0, 
    ('am', 'tired'): 1.0, 
    ('tired', '.'): 0.5, 
    ('you', 'in'): 0.4, 
    ('in', 'england'): 0.5, 
    ('england', '?'): 1.0, 
    ('<BOS>', 'were'): 0.16666666666666666, 
    ('were', 'you'): 1.0, 
    ('in', 'mexico'): 0.5, 
    ('mexico', '?'): 1.0
})
"""

# @model: the dictionary containing your model
# @model.lm: the name of the file you want to save to
print("Saving model...")
print("Saved %d bigrams from train.\n" % sum([len(i) for i in model.items()]))
pickle.dump(dict(model), open('bigram_model.lm', 'wb'))


# get perplexity of training corpus (unigrams vs bigrams)
train_uni_perp, train_bi_perp = get_perplexity(uni_counter, model)
print(f"Train set unigram perplexity: {train_uni_perp}") # 3.87081990216204
print(f"Train set bigram perplexity: {train_bi_perp}\n") # 1.55579776754989



##########################################################################################
### TEST #################################################################################
### Q: Which of the test sentences is the most probable, according to our model?
### We can answer this Q by calculating perplexity of our unigram and bigram models. 
### (Perplexity is one measure of how likely a given language model will predict the test data)
##########################################################################################
print("Reading test data...\n")
with open('test.txt', 'r') as file:
    test = file.readlines()
new_test = preprocess(test)

# create unigrams and bigrams, plus their respective counters
print("Calculating test bigram probabilities...\n")
test_unigrams = [token for line in new_test for token in line.split(" ")]
test_uni_counter = Counter(test_unigrams)
print(f"Test Unigrams: {test_uni_counter}\n")
"""
Test Unigrams: Counter({
    '<BOS>': 5, 
    'you': 4, 
    '?': 4, 
    'in': 4, 
    'are': 3, 
    'mexico': 3, 
    'where': 1, 
    'were': 1, 
    'england': 1, 
    'i': 1, 
    'am': 1, 
    '.': 1, 
    'still': 1, 
    '': 1})
"""

test_bigrams = make_bigrams(new_test)
test_bi_counter = Counter(test_bigrams)
print(f"Test Bigrams: {test_bi_counter}\n")
print("Saved %d bigrams from test.\n" % sum([len(i) for i in test_bi_counter.items()]))
"""
Test Bigrams: Counter({
    ('are', 'you'): 3, 
    ('in', 'mexico'): 3, 
    ('you', 'in'): 2, 
    ('<BOS>', 'are'): 2, 
    ('mexico', '?'): 2, 
    ('<BOS>', 'where'): 1, 
    ('where', 'are'): 1, 
    ('you', '?'): 1, 
    ('<BOS>', 'were'): 1, 
    ('were', 'you'): 1, 
    ('in', 'england'): 1, 
    ('england', '?'): 1, 
    ('<BOS>', 'i'): 1, 
    ('i', 'am'): 1, 
    ('am', 'in'): 1, 
    ('mexico', '.'): 1, 
    ('you', 'still'): 1, 
    ('still', 'in'): 1, 
    ('?', ''): 1
})

Saved 38 bigrams from test.
"""

# probability & log probability for each sentence in test 
results = []
for sent in new_test:
    sent_bigrams = []
    sent = sent.split(" ")
    for i in range(len(sent)):
        if i!=0:
            sent_bigrams.append((sent[i-1], sent[i]))
    sent_prob = 1.0
    for bigram in sent_bigrams:
        if bigram in model:
            sent_prob *= model[bigram]
        else:
            sent_prob += 0.0
    results.append((math.log(sent_prob), sent_prob, sent))
print(f"log prob and probabilities for each sentence in test data:\n {results}\n")
"""
[(-3.4011973816621555, 0.03333333333333333, ['<BOS>', 'where', 'are', 'you', '?']), 
(-3.4011973816621555, 0.03333333333333333, ['<BOS>', 'were', 'you', 'in', 'england', '?']), 
(-2.3025850929940455, 0.1, ['<BOS>', 'are', 'you', 'in', 'mexico', '?']), 
(-2.4849066497880004, 0.08333333333333333, ['<BOS>', 'i', 'am', 'in', 'mexico', '.']), 
(-2.995732273553991, 0.05, ['<BOS>', 'are', 'you', 'still', 'in', 'mexico', '?', ''])]
"""

# normalize each bigram by the frequency of the first unigram
test_bi_dict = defaultdict(lambda : defaultdict(float)) 
for bigram in test_bi_counter: # Calculate probabilities
    test_bi_dict[bigram] = test_bi_counter[bigram] / test_uni_counter[bigram[0]]
print(f"Test bigram dict: {test_bi_dict}\n")
"""
Test bigram dict: defaultdict(<function <lambda> at 0x7fc6f01f4310>, {
    ('<BOS>', 'where'): 0.2, 
    ('where', 'are'): 1.0, 
    ('are', 'you'): 1.0, 
    ('you', '?'): 0.25, 
    ('<BOS>', 'were'): 0.2, 
    ('were', 'you'): 1.0, 
    ('you', 'in'): 0.5, 
    ('in', 'england'): 0.25, 
    ('england', '?'): 1.0, 
    ('<BOS>', 'are'): 0.4, 
    ('in', 'mexico'): 0.75, 
    ('mexico', '?'): 0.6666666666666666, 
    ('<BOS>', 'i'): 0.2, 
    ('i', 'am'): 1.0, 
    ('am', 'in'): 1.0, 
    ('mexico', '.'): 0.3333333333333333, 
    ('you', 'still'): 0.25, 
    ('still', 'in'): 1.0, 
    ('?', ''): 0.25
})
"""

print("Evaluating model on test corpus...\n")
model = pickle.load(open('bigram_model.lm', 'rb'))
### comment: we expect the bigram model to have lower perplexity than unigram model

# # get perplexity of test corpus (unigrams vs bigrams)
test_uni_perp, test_bi_perp = get_perplexity(test_uni_counter, model)
print(f"Test set unigram perplexity: {test_uni_perp}") # 3.64690859356563
print(f"Test set bigram perplexity: {test_bi_perp}\n") # 1.6008005239652336