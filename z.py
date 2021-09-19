import pickle
import numpy as np
import sys

with open('./embeddings/w2.pkl', 'rb') as f:
    w2 = pickle.load(f)
with open('./embeddings/word2ind.pkl', 'rb') as f:
    word2ind = pickle.load(f)
with open('./embeddings/ind2word.pkl', 'rb') as f:
    ind2word = pickle.load(f)

W2 =np.array(w2)
# print(word2ind)

# returns top10 nearest neighbours to 'str' string
def top10(str):
  array = []
  v1 = W2.T[word2ind[str]]
  for i in range (0,len(W2.T)):
    v2 = W2.T[i]
    angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    array.append([angle,ind2word[i]])
  
  array.sort(reverse=True)
  return array[:10]

if __name__ == '__main__':
  
  if(len(sys.argv) != 2):
    print("Usage: python3 z.py <word>")
  else:
    word = sys.argv[1]
    print(top10(word))