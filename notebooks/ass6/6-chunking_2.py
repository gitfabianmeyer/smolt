import nltk
from nltk.chunk.regexp import RegexpChunkRule

## Grammar section
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

# NP chunking rules
grammar = """NP: 
                {<NN>}

"""

cp = nltk.RegexpParser(grammar)

result = cp.parse(sentence)
print (result)
