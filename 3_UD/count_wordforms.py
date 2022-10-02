"""
L645 practical 9/6/22

INSTRUCTIONS:
1. Download a UD dataset from https://universaldependencies.org/ 
        (/Users/lilykawaoto/IU_L645/3_UD/aqz_tudet-ud-test.conllu)
2. Get a count / percentage of how many words have more than one analysis.
3. Plot frequency vs analyses

Type ambiguity = what % of types (unique tokens) have >1 analysis?
Token ambiguity = what % of (all) tokens have >1 analysis?

*** CONLLU column descriptions ***
1. ID: Word index, integer starting at 1 for each new sentence; may be a range for tokens with multiple words.
2. FORM: Word form or punctuation symbol.
3. LEMMA: Lemma or stem of word form.
4. UPOSTAG: Universal part-of-speech tag drawn from our revised version of the Google universal POS tags.
5. XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
6. FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
7. HEAD: Head of the current token, which is either a value of ID or zero (0).
8. DEPREL: Universal Stanford dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
9. DEPS: List of secondary dependencies (head-deprel pairs).
10. MISC: Any other annotation.

"""

import sys
import matplotlib
import matplotlib.pyplot as plt

# use dictionary to keep track of number of occurrences
words = {}
frequencies = {}
analyses = {}

n_tokens = 0
n_types_multiple_analyses = 0
n_tokens_multiple_analyses = 0
n_types = 0
n_analyses = 0

# take in a .conllu file as command line argument
for line in open(sys.argv[1]).readlines():
    line = line.strip()
    if line == '':
        continue
    if line[0] == '#':
        continue
    row = line.split('\t')
    if '.' in row[0] or '-' in row[0]:
        continue

    wordform = row[1]
    analysis = row[2] + '/' + row[3] + '/' + row[5] # lemma/POS/list of features

    if wordform not in frequencies: 
        frequencies[wordform] = 0
    frequencies[wordform] += 1
    
    if wordform not in analyses: 
        analyses[wordform] = []
    analyses[wordform].append(analysis)
    n_tokens += 1

for wordform in analyses:
    if len(set(analyses[wordform])) > 1:
        n_types_multiple_analyses += 1.0

for wordform in frequencies:
    if frequencies[wordform] > 1:
        n_tokens_multiple_analyses += 1.0

unique_wordforms = list(frequencies.keys())
print(f"List of unique wordforms: {unique_wordforms}\n")

n_types = len(unique_wordforms)
print(f"Number of types (ie, number of unique wordforms): {n_types}\n")

perc_type_mult_analyses = n_types_multiple_analyses/n_types
print(f"Percentage of ambiguous wordforms (ie, has multiple analyses): {perc_type_mult_analyses}\n")

perc_ambig_tokens = n_tokens_multiple_analyses/n_tokens
print(f"Percentage of ambiguous tokens: {perc_ambig_tokens}\n")


# plot frequency vs analyses (x-axis freq, y-axis analyses, dot word)

word_list = [word for word in frequencies]
frequency_list = [frequencies[word] for word in frequencies]
num_analyses_list = [len(set(analyses[word])) for word in frequencies]

fig, ax = plt.subplots()
# ax.scatter(frequency_list, num_analyses_list)
ax.scatter(num_analyses_list, frequency_list)

for i,word in enumerate(frequencies):
    ax.annotate(word, (num_analyses_list[i], frequency_list[i]))

plt.xlabel("frequency")
plt.ylabel("num_analyses")
plt.axis([0, 4.0, 0, 20.0]) # zoom in to focus on concentrated area
plt.show()