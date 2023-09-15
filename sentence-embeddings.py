import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import time
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_name", type=str, help="HF model name (ex: all-mpnet-base-v2, all-distilroberta-v1, all-MiniLM-L6-v2, roberta-base-nli-stsb-mean-tokens)")
    parser.add_argument("-txt_file", default=[], nargs='+', help="files with sentences to build embeddings")
    args = parser.parse_args()
    np.random.seed(1234)

tic = time.time()
model = SentenceTransformer(args.model_name)
print('Read model {}, took {:.2f} seconds'.format(args.model_name, time.time()-tic))

for txt_file in args.txt_file:
    tic = time.time()
    with open(txt_file, 'r') as fd:
        sentences = [l.strip() for l in fd]

    embeddings = model.encode(sentences)
    np.save('{}.{}.npy'.format(txt_file, args.model_name), embeddings)
    print('Encoded sentences {}, took {:.2f} seconds ({})'.format(embeddings.shape, time.time()-tic, txt_file))



