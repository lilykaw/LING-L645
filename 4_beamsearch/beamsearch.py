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

    # Loop over time
    for t in range(max_T):
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            # Loop over possible alphabet outputs
            for c in range(max_A - 1):
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

### print result
output = []
for i, seq in enumerate(result):
    for pred in seq[0]:
        output.append(alphabet[pred])
prediction = ""
for i,char in enumerate(output):
     if i==0 or char!=output[i-1]:
        prediction += char

print(prediction)

s = sns.heatmap(logits, annot=False, cbar=False, cmap="Blues")
s.set(xlabel='predicted alphabet char', ylabel='logits')
plt.xticks(np.arange(len(alphabet))+0.5, alphabet)
plt.show()

# "we must find a new home in the stars"


### apply CTCdecoder_diff file to improve this beamsearch algorithm using CTC 
### (connectionist temporal classification)
# CTC summary: https://distill.pub/2017/ctc/
# CTC implementation example 1: https://gist.githubusercontent.com/awni/56369a90d03953e370f3964c826ed4b0/raw/35b99ac85c5b4cfeb75682f059d3e876fe3a7d53/ctc_decoder.py
# CTC implementation example 2: https://github.com/githubharald/CTCDecoder 