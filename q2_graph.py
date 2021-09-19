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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

list_of_closest_words = {
    "good": ["good", "alignment", "manually", "newness", "wildly", "resize", "earcup", "traveled", "animated", "authorized"],
    "device": ["device", "soaked", "extraordinary", "march", "shall", "toyed", "tourist", "sennheisers", "read", "beatles"],
    "terrible": ["terrible", "stars", "channels", "1102", "forecasts", "enviornment", "grounding", "pittsburgh", "bar", "ounce"],
    "hard": ["hard", "solve", "also", "attributes", "800", "recoton", "frozen", "forecasts", "xa", "stars"],
    "awesome": ["awesome", "copyright", "discriminating", "cad", "500k", "pricier", "solo", "diaphragm", "tokyo", "deter"],
    "camera": ["camera", "ward", "forte", "camcorders", "garmin", "assembly", "fuses", "teh", "165", "refunding"]
}

global_embeddings = []
global_annotations = []
for w in list_of_closest_words:

    tsne = TSNE(n_components=2)

    w_embeddings = [W2.T[word2ind[w]]]
    w_embeddings += [W2.T[word2ind[cw]] for cw in list_of_closest_words[w]]
    global_embeddings += w_embeddings

    annotations = [w]
    annotations += [cw for cw in list_of_closest_words[w]]
    global_annotations += annotations

    flattened_embeddings = tsne.fit_transform(w_embeddings)

    X_coordinates = flattened_embeddings[:,0].tolist()
    Y_coordinates = flattened_embeddings[:,1].tolist()

    plt.figure()
    plt.title(f"Closest words for {w}")
    plt.scatter(X_coordinates, Y_coordinates)

    for w_no, annotation in enumerate(annotations):
        plt.annotate(annotation, (X_coordinates[w_no], Y_coordinates[w_no]))

    plt.savefig(f"./images2/{w}.png")


#  all words combined
tsne = TSNE(n_components=2)

flattened_embeddings = tsne.fit_transform(global_embeddings)
X_coordinates = flattened_embeddings[:,0].tolist()
Y_coordinates = flattened_embeddings[:,1].tolist()

plt.figure()
plt.title("Combined")
plt.scatter(X_coordinates, Y_coordinates)

for w_no, annotation in enumerate(global_annotations):
    plt.annotate(annotation, (X_coordinates[w_no], Y_coordinates[w_no]))

plt.savefig("./images2/all_words_combined.png")