from numpy import log10, sqrt, array, mean, linalg, average, arange, argmin, isin, sum as np_sum, zeros, diag
from indices import InvertedIndex, PositionalIndex, SubInvertedIndex
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds


def intersect(P1,P2):
    '''
    Intersecting two posting lists P1 and P2 when they are sorted increasingly in order to have a linear
    (wrt the sum of the posting's length) complexity.
    '''
    p1,p2=0,0
    n1,n2=len(P1),len(P2)
    result=[]
    while p1<n1 and p2<n2:
        if P1[p1]==P2[p2]:
            result.append(P1[p1])
            p1+=1
            p2+=1
        elif P1[p1]<P2[p2]:
            p1+=1
        else:
            p2+=1
    return result


def intersect_increasing_freq(P):
    '''
    Intersecting many posting lists in the list P starting from the postings of the rarest terms (in terms of document frequency).
    The reason why we sort the postings according to their length is that as a rule of thumb the intersection of small postings
    is more likely to yield smaller postings than the intersection of long postings.
    '''
    # sorting the elements of P by length
    _ , L = zip(*sorted(zip([len(p) for p in P], P)))
    L=list(L)
    rest=L[1:]
    result=L[0]
    while len(rest)>0 and len(result)>0:
        posting=rest[0]
        result=intersect(result,posting)
        rest=rest[1:]
    return result

def union_two_postings(P1,P2):
    '''
    Merging (union) two posting lists P1 and P2 when they are sorted increasingly in order to have a linear
    (wrt the sum of the posting's length) complexity.
    '''
    p1,p2=0,0
    n1,n2=len(P1),len(P2)
    result=[]
    while p1<n1 and p2<n2:
        if P1[p1]==P2[p2]:
            result.append(P1[p1])
            p1+=1
            p2+=1
        elif P1[p1]<P2[p2]:
            result.append(P1[p1])
            p1+=1
        else:
            result.append(P2[p2])
            p2+=1
    while p1<n1:
        result.append(P1[p1])
        p1+=1
    while p2<n2:
        result.append(P2[p2])
        p2+=1
    return result

def union_many_postings(P):
    '''
    Merging many postings 
    '''
    rest=P[1:]
    result=P[0]
    while len(rest)>0:
        result = union_two_postings(result, rest[0])
        rest=rest[1:]
    return result


def positional_intersect_two(P1,P2, k, sort):
    '''
    Intersecting two positional postings P1 and P2 of two terms.
    P1 and P2 should be of the form [{DocID1:[raw_freq, [position1,position2,...]]}, {DocID2:[raw_freq, [position1,position2,...]]},... ]
    or {DocID1:[raw_freq, [position1,position2,...]], DocID2:[raw_freq, [position1,position2,...]],...} 
    k is an integer which says how far the term t2 (of posting P2) is away from t1 (of posting P1). 
    Careful: the order of P1 and P2 matters: t1 comes first then t2.
    '''
    p1,p2=0,0
    if not sort: 
        n1,n2=len(P1.keys()), len(P2.keys())
        P1, P2=sorted(P1.items(), key=lambda item: item[0]), sorted(P2.items(), key=lambda item: item[0])
        result=dict()
    else: n1,n2=len(P1),len(P2); result=[]
        
    while p1<n1 and p2<n2: 
        if sort: docID1, docID2 = list(P1[p1].keys())[0], list(P2[p2].keys())[0]   
        else: docID1, docID2 = P1[p1][0], P2[p2][0]
        if docID1==docID2:
            l=[]
            # lists of terms positions in the documents
            if sort: l1, l2=list(P1[p1].values())[0][1], list(P2[p2].values())[0][1]  
            else: l1,l2=P1[p1][1][1], P2[p2][1][1]
            pp1=0
            nn1,nn2=len(l1), len(l2)
            while pp1<nn1:
                pp2=0
                while pp2<nn2:
                    if 1<=l2[pp2]-l1[pp1]<=k:
                        # the term t2 is in k words ahead from the term t1 (after preprocessing the document)
                        l.append(l2[pp2])
                    elif l2[pp2]>l1[pp1]+k: 
                        break
                    pp2+=1
                pp1+=1
            if len(l)>0:
                # To make sure that only documents that match the phrase "t1 t2" (after preprocessing) are taken into account
                if sort: result.append({docID1:[len(l), l]}) # useful for first version
                else: result[docID1]=[len(l), l]
            p1+=1
            p2+=1
        elif docID1<docID2:
            p1+=1
        else: p2+=1
    return result



def positional_intersect(P, k, sort):
    '''
    Intersecting positional posting lists, returning the IDs of the relevant documents
    '''
    result=P[0]
    rest=P[1:]
    while len(rest)>0 and len(result)>0:
        result=positional_intersect_two(result, rest[0], k, sort)
        rest=rest[1:]
    if sort: return [list(dico.keys())[0] for dico in result] 
    return result.keys()



def idf(index, formula='idf'):
    '''
    Computing the inverted document frequency (idf) of the index terms
    '''
    if type(index)==PositionalIndex:
        if formula=='idf':
            return {term: log10(len(index.documents)/index.index[term][0]) for term in index.index}
        elif formula=='prob_idf':
            return {term: log10((len(index.documents)-index.index[term][0])/index.index[term][0]) for term in index.index}
        
    elif type(index)==InvertedIndex:
        if formula=='idf':
            return {term: log10(len(index.documents)/len(index.index[term])) for term in index.index}
        elif formula=='prob_idf':
            return {term: log10((len(index.documents)-len(index.index[term]))/len(index.index[term])) for term in index.index}
        
    else: raise TypeError ("idf function only supports PositionalIndex or InvertedIndex types")
    

def tf(index, formula='logarithm'):
    '''
    Computing the term frequencies
    '''
    if formula not in ["boolean", "logarithm", "raw"]:
        raise ValueError ("formula argument should be in ['boolean', 'logarithm', 'raw']")

    if type(index)==PositionalIndex:
        terms=index.index.keys()
        if formula=="boolean":
            if index.sort_postings:
                return {ID: {term: (term_info[0]>0)*1 for term in terms for doc in index.index[term][1:] for doc_id, term_info in doc.items() if doc_id==ID} for ID in index.documents}
            else:
                return {ID: {term: (term_info[0]>0)*1 for term in terms for doc_id, term_info in index.index[term][1].items() if doc_id==ID} for ID in index.documents}
        elif formula=="raw":
            if index.sort_postings:
                return {ID: {term: term_info[0] for term in terms for doc in index.index[term][1:] for doc_id, term_info in doc.items() if doc_id==ID} for ID in index.documents}
            else:
                return {ID: {term: term_info[0] for term in terms for doc_id, term_info in index.index[term][1].items() if doc_id==ID} for ID in index.documents}
        elif formula=="logarithm":
            if index.sort_postings:
                return {ID: {term: 1+log10(term_info[0]) for term in terms for doc in index.index[term][1:] for doc_id, term_info in doc.items() if doc_id==ID} for ID in index.documents}
            else:
                return {ID: {term: 1+log10(term_info[0]) for term in terms for doc_id, term_info in index.index[term][1].items() if doc_id==ID} for ID in index.documents}
    
    elif type(index)==InvertedIndex:
        if formula=="logarithm":
            return {ID: {term: 1+log10(index.raw_freq[ID][term]) for term in index.raw_freq[ID]} for ID in index.documents}
        elif formula=="raw":
            return {ID: {term: index.raw_freq[ID][term] for term in index.raw_freq[ID]} for ID in index.documents}
        elif formula=="boolean":
            return {ID: {term: 1 for term in index.raw_freq[ID]} for ID in index.documents}
    
    else: raise TypeError ("tf function only supports PositionalIndex or InvertedIndex types")
        
        
def unigram(index):
    '''
    Computing the unigram language model 
    '''
    if type(index)!=InvertedIndex:
        raise TypeError ("unigram_model supports only an InvertedIndex type")
    N={term: sum([index.raw_freq[id][term] for id in index.index[term]])
      for term in index.index}
    sum_N = sum(N.values())
    return {ID: {term: N[term]/sum_N for term in index.raw_freq[ID]} for ID in index.documents}        
        
        
def doc_embed(index, tfreqs, idfs, word_embeds):
    '''
    Gives the embedding of the documents in the collection using tf-idf weighted words embeddings
    word embeddings should be dictionary like {term: embedding}
    '''
    if type(index)!=InvertedIndex:
        raise TypeError ("doc_embed supports only an InvertedIndex type")

    return {ID: average(a=[word_embeds[term] for term in index.raw_freq[ID] if term in word_embeds], axis=0,
                        weights=[tfreqs[ID][term]*idfs[term] for term in index.raw_freq[ID] if term in word_embeds]) 
            for ID in tqdm(index.documents)}    


def simple_plot(x,y,title,xlabel,ylabel):
    '''
    '''
    plt.title(title)
    plt.plot(x,y, 'o-')
    plt.xlabel(xlabel)
    plt.xticks(x)
    plt.ylabel(ylabel)
    plt.show()
    

def cluster_docs(index, doc_embeds, num_min_centr=10,num_max_centr=100, step=10, penalty=0.1, plot=False):
    '''
    Clustering documents using KMEANS++, documents are represented by their embeddings.
    The optimal number of cluster is computed here.
    '''
    X=[doc_embeds[ID] for ID in index.documents]
    # weighting the documents using their normalized length
    doc_weights=array([len(index.raw_freq[ID]) for ID in index.documents])
    doc_weights=doc_weights/sum(doc_weights)
    inertias=[]
    assert num_min_centr<num_max_centr
    ks=arange(num_min_centr,num_max_centr,step)
    kmeans=[]
    print("Computing K-means model for different values of K")
    for k in tqdm(ks):
        kmeans.append(KMeans(n_clusters=k, random_state=2, init='k-means++').fit(X, sample_weight=doc_weights))
        inertias.append(kmeans[-1].inertia_)
    # optimal choice of k by minimizing obj(k)=inertia(k)-penalty*k wrt k
    obj_k = array(inertias)-penalty*ks
    if plot:
        simple_plot(ks, obj_k, title=f"Penalised Inertia = Inertia - {penalty} * number_cluster", 
        xlabel="number of clusters", ylabel='Penalised Inertia')
    opt_kmeans=kmeans[argmin(obj_k)]
    return opt_kmeans 


def opt_svd(mat, min_k, max_k, step_k=10, penalty=1, plot=False):
    '''
    Computing the "optimal" SVD decompostion of a matrix mat maximizing a certain objective
    We advice to take a high enough penalty value so to avoid memory issues
    ''' 
    r=min(mat.shape[1], mat.shape[0])-1 # approximately the maximum rank of mat (-1 used for the scipy solver to work)
    assert min_k<max_k
    ks=arange(min_k, max_k, step_k)
    print("Computing SVDs for different values of k")
    for k in tqdm(ks):
        if r<k:
            raise ValueError ('The number of singular values must not be greater than the maximal rank of the matrix.\nSee https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html')        
        if k==ks[0]:
            usv_opt=svds(mat,k=ks[0])
            opt_k=ks[0]
            obj_k=[sum(usv_opt[1])-penalty*ks[0]]
        else:
            u,s,v = svds(mat,k=k)
            obj_k.append(sum(s)-penalty*k)
            if sum(usv_opt[1])-penalty*opt_k < obj_k[-1]:
                usv_opt=u,s,v
                opt_k=k
    if plot:
        simple_plot(x=ks, y=obj_k, title=f"Penalised sum of singular values (PSSV) = SSV - {penalty} * number of SV",
                    xlabel="number of singular values", ylabel='PSSV')
    return usv_opt


def wd(index):
    '''
    Builds word-document matrix of occurences, along with the word to index (wtoi) and 
    document to index (dtoi) associated, wtoi and dtoi are used to identify terms and documents
    in the word-document matrix
    '''
    if type(index)not in [InvertedIndex, SubInvertedIndex]:
        raise TypeError ("wd supports only an (Sub)InvertedIndex types")

    WD=zeros((len(index.index), len(index.documents)))
    T=list(index.index)
    wtoi={term: i for i, term in enumerate(T)}
    # itow={v:k for k,v in wtoi.items()}
    D=list(index.documents)
    dtoi={ID: i for i,ID in enumerate(D)}
    # itod={v:k for k in dtoi.items()}
    for pos,d in tqdm(enumerate(D)):
        WD[[wtoi[term] for term in index.raw_freq[d]],pos]=\
        [index.raw_freq[d][term] for term in index.raw_freq[d]]

    return WD, wtoi, dtoi


def rank_documents(index, doc_scores, top):
    top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
    if len(top_documents)>0:
        if top_documents[0][1]==0:
            return []
    return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]