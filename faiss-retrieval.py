import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import time
import faiss
import argparse
import numpy as np

def read_vectors(f, normalize_L2=True):
    tic = time.time()
    with open(f, 'rb') as fd:
        x = np.load(f)
    if normalize_L2:
        faiss.normalize_L2(x=x)
    print('Read vectors {}, took {:.2f} seconds ({})'.format(x.shape, time.time()-tic, f))
    return x

def create_index(x, index_type, nlist, gpu):
    tic = time.time()
    n, d = x.shape
    if index_type == 'FlatIP':
        index = faiss.IndexFlatIP(d)
    elif index_type == 'IVFFlat':
        nlist = 100
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        assert not index.is_trained
        index.train(x)
    if gpu:
        res = faiss.StandardGpuResources() # use a single GPU
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
        index_gpu.add(x)
    else:
        index.add(x)
    print('Created index {}, took {:.2f} seconds'.format(x.shape, time.time()-tic))
    return index_gpu if gpu else index

def query_index(index, x, k):
    tic = time.time()
    D, I = index.search(x, k)
    print('Search query {}, took {:.2f} seconds'.format(x.shape, time.time()-tic))
    return D, I

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("emb_index", type=str, help="embeddings to create index")
    parser.add_argument("txt_index", type=str, help="texts corresponding to index")
    parser.add_argument("-emb_query", default=[], nargs='+', help="embeddings to query the index")    
    parser.add_argument("-k", default=10, type=int, help="number of nearest neighbours (10)")
    parser.add_argument('-normalize_L2', action='store_true', help='normalize vectors using L2 norm before adding to index, use with cosine similarity distance')
    parser.add_argument("-itype", default="FlatIP", help="index type (FlatIP, IVFFlat)")
    parser.add_argument("-nlist", default=0, type=int, help="split vectors into nlist clusters, according to the quantizer (0)")
    parser.add_argument("-nprobe", default=0, type=int, help="during search, only nprobe clusters are searched (0)")
    parser.add_argument('-gpu', action='store_true', help='use gpu')
    parser.add_argument("-sep", default=None, help="similars separator")
    args = parser.parse_args()
    np.random.seed(1234)

    x = read_vectors(args.emb_index, normalize_L2=args.normalize_L2)
    index = create_index(x, args.itype, args.nlist, args.gpu)
        
    with open(args.txt_index, 'r') as fd:
        txt = [l.strip() for l in fd]

    for emb_query in args.emb_query:
        y = read_vectors(emb_query, normalize_L2=args.normalize_L2)
        D, I = query_index(index, y, args.k+1)        
        print('Dumping results... ({}.? OR .k{})'.format(emb_query,args.k))
        
        if args.sep is not None:
            fds = open('{}.k{}'.format(emb_query, args.k), 'w')
        else:
            fds = [open('{}.{}'.format(emb_query, k), 'w') for k in range(args.k)]
            
        for i in range(I.shape[0]):
            vec = []
            for k in range(args.k): #0 .. k-1
                pos = I[i,k]
                score = D[i,k]
                sent = txt[pos]
                if args.sep:
                    vec.append(args.sep+" "+sent)
                else:
                    print("{}:{:.5f}\t{}".format(pos,score,sent), file=fds[k])
            if args.sep is not None:
                print(" ".join(vec), file=fds)
