import nltk
from nltk.chunk.regexp import RegexpChunkRule

## Grammar section
sentence = [("Barack", "NNP"), ("Obama", "NNP"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("the","DT"), ("state","NN"), ("Hawaii", "NNP")]

# NP chunking rules
grammar = """NP: 
                {<NN>}
"""

cp = nltk.RegexpParser(grammar)

result = cp.parse(sentence)
print (result)
