# Practical 4: beamsearch with CTC, plus minor improvements 
# (discard low probability cells, avoid beam expansion until a certain threshold is reached)
# https://cl.indiana.edu/~ftyers/courses/2022/Autumn/L-645/practicals/beamsearch/beamsearch.html


from math import log
import numpy as np
import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

### beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence

    max_T, max_A = data.shape

    start_expanding = False

    # Loop over time
    for t in range(max_T):
        all_candidates = list()
        
        # avoid beam expansion at the beginning until the probability reaches
        # a threshold of 0.999
        if data[t, 0] < 0.9999:
            start_expanding = True
        if not start_expanding:
            continue

        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            # Loop over possible alphabet outputs
            for c in range(max_A - 1):

                # discard low probability cells in data matrix
                if data[t, c] < 0.00001:
                    continue
                
                candidate = [seq + [c], score - log(data[t, c])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

### define a sequence of 10 words (rows) over a vocab of 5 words (columns), 
# e.g.
#      a  bites cat  dog  the
# 1   0.1  0.2  0.3  0.4  0.5
# 2   0.5  0.3  0.5  0.2  0.1
# ...
# 10  0.3  0.4  0.5  0.2  0.1 

data = [[0.1, 0.2, 0.3, 0.4, 0.5],
        [0.4, 0.3, 0.5, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.2, 0.1]]

data = np.array(data)

beam_width = 5


### load output.json
with open("/Users/lilykawaoto/IU_L645/4_beamsearch/output.json", 'r') as f:
    file = json.load(f)

alphabet = file["alphabet"]
logits = np.array(file["logits"])


### decode sequence
# result = beam_search_decoder(data, beam_width) # provided dummy data
result = beam_search_decoder(logits, beam_width)
# print(result)

### print result
output = []
for i, seq in enumerate(result):
    for pred in seq[0]:
        output.append(alphabet[pred])
    output.append("\n")
prediction = ""
for i,char in enumerate(output):
     if i==0 or char!=output[i-1]:
        prediction += char

print(prediction)

# s = sns.heatmap(logits, annot=False, cbar=False, cmap="Blues")
# s.set(xlabel='predicted alphabet char', ylabel='logits')
# plt.xticks(np.arange(len(alphabet))+0.5, alphabet)
# plt.show()

# "we must find a new home in the stars"