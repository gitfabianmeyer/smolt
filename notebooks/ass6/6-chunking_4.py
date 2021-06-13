import sys
import nltk
from nltk.chunk.regexp import RegexpChunkRule


# helper functions for tuple creation
def group(lst, n):
  for i in range(0, len(lst), n):
    val = lst[i:i+n]
    if len(val) == n:
      yield tuple(val)

def postag(lst):
  for i in range(0, len(lst), 3):
    val = lst[i:i+2]
    if len(val) == 2:
      yield tuple(val)


# open file
raw_annotations = open("train.small.data").read()
split_annotations = raw_annotations.split()


# create tuples of gold annotation and postagged text
reference_annotations = list(group(split_annotations, 3))

postagged_text = list(postag(split_annotations))



## Grammar section

# NP chunking rules
grammar = """NP: 
                {<NNP>+}
                {<DT><NN>}
                {<VBD><NN>}
"""
cp = nltk.RegexpParser(grammar)

result = cp.parse(postagged_text)


# Convert prediction to multiline string and then to list (includes pos tags)
multiline_string = nltk.chunk.tree2conllstr(result)
listed_pos_and_np = multiline_string.split()


formatted_prediction = list(group(listed_pos_and_np, 3))

# output tab-separated result, add gold annotation
for n,res in enumerate(formatted_prediction):
  print (res[0] + "\t" +res[1] + "\t" + reference_annotations[n][2] + "\t" +res[2])



