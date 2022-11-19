# Fran's example code

import sys, math, re, pickle
from collections import defaultdict, Counter

def tokenise(s):
    """Tokenise a line"""
    o = re.sub('([^a-zA-Z0-9\']+)', ' \g<1> ', s.strip())
    return re.sub('  *', ' ', o).split(' ')

model = defaultdict(lambda : defaultdict(float)) 

bigrams, unigrams = defaultdict(Counter), Counter() # Unigram and bigram counts 

line = sys.stdin.readline()
while line: # Collect counts from standard input
    tokens = ['<BOS>'] + tokenise(line)
    for i in range(len(tokens) - 1):
        bigrams[tokens[i]][tokens[i+1]] += 1
        unigrams[tokens[i]] += 1
    line = sys.stdin.readline()
# for sys.stdin.readline(): hit Ctrl+D to signal end of input stream in Terminal

print(f"Unigrams: {unigrams}")
print(f"Bigrams: {bigrams}")
"""
### Bigrams: 
defaultdict(<class 'collections.Counter'>, {
    '<BOS>': Counter({'are': 3, 'where': 1, 'i': 1, 'were': 1}), 
    'are': Counter({'you': 4}), 
    'you': Counter({'in': 2, 'still': 1, '?': 1, 'tired': 1}), 
    'still': Counter({'here': 1}), 
    'here': Counter({'?': 1}), 
    '?': Counter({'': 5}), 
    'where': Counter({'are': 1}), 
    'tired': Counter({'?': 1, '.': 1}), 
    'i': Counter({'am': 1}), 
    'am': Counter({'tired': 1}), 
    '.': Counter({'': 1}), 
    'in': Counter({'england': 1, 'mexico': 1}), 
    'england': Counter({'?': 1}), 
    'were': Counter({'you': 1}), 
    'mexico': Counter({'?': 1})
})
"""
for i in bigrams: # Calculate probabilities
    for j in bigrams[i]:
        model[i][j] = bigrams[i][j] / unigrams[i]

print(model)
"""
defaultdict(<function <lambda> at 0x7fe44013b670>, {
    '<BOS>': defaultdict(<class 'float'>, {'are': 0.5, 'where': 0.16666666666666666, 'i': 0.16666666666666666, 'were': 0.16666666666666666}), 
    'are': defaultdict(<class 'float'>, {'you': 1.0}), 
    'you': defaultdict(<class 'float'>, {'still': 0.2, '?': 0.2, 'tired': 0.2, 'in': 0.4}), 
    'still': defaultdict(<class 'float'>, {'here': 1.0}), 
    'here': defaultdict(<class 'float'>, {'?': 1.0}), 
    '?': defaultdict(<class 'float'>, {'': 1.0}), 
    'where': defaultdict(<class 'float'>, {'are': 1.0}), 
    'tired': defaultdict(<class 'float'>, {'?': 0.5, '.': 0.5}), 
    'i': defaultdict(<class 'float'>, {'am': 1.0}), 
    'am': defaultdict(<class 'float'>, {'tired': 1.0}), 
    '.': defaultdict(<class 'float'>, {'': 1.0}), 
    'in': defaultdict(<class 'float'>, {'england': 0.5, 'mexico': 0.5}), 
    'england': defaultdict(<class 'float'>, {'?': 1.0}), 
    'were': defaultdict(<class 'float'>, {'you': 1.0}), 
    'mexico': defaultdict(<class 'float'>, {'?': 1.0})
})
"""
print('Saved %d bigrams.' % sum([len(i) for i in model.items()]))
pickle.dump(dict(model), open('model.lm', 'wb'))