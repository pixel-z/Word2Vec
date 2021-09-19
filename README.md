# Word2Vec

All the top 10 vectors closest to 5 different words are in Report.pdf


### q1 - Co-Occurrence Matrix by applying Singular Value Decomposition
Run the jupyter file `q1.ipynb` top to bottom.  

Print functions are left in understanding the code and its structure.

Embeddings, pre-trained model and graph images for q1 saved:
https://drive.google.com/drive/folders/1D7hJuZ1_D6s4t8gjbubbGI03YFin5PC8

The training time is quite low so it is not really practical to import pre-trained model instead the whole thing.

### q2 - CBOW Model
Run the jupyter file `q2.ipynb` top to bottom.

Embeddings, pre-trained model and graph images for q2 saved:
https://drive.google.com/drive/folders/1AOnJC7WR35HOzVfp5KrtA7J5bFF5LrTO

The embeddings for CBOW are also attached in the folder `embeddings`.
To check the `top10` for any word we can make use of the script `z.py`.

Restoring of pre-trained model is all shown in `z.py` and can be used.

E.g:    
```bash
# top10 for `camera`, run:
python3 z.py camera

# top10 for `awesome`, run:
python3 z.py awesome

# similarly for others
```