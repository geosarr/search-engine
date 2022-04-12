from tqdm import tqdm
from numpy import array, where


def inv_rank_most_relevant_doc(model, query, judgments, at_K):
    '''
    Finds the inverse rank of the most relevant document
    '''
    in_rank_most_rel_doc_id=0
    most_rel_doc_id=sorted(judgments[query].items(), key=lambda item: item[1], reverse=True)[0][0]
    model.top=at_K 
    results=model.retrieval(query)
    retrieved_docs=array(results)[:,0] # IDs of the retrieved docs
    if len(retrieved_docs)>0:
        rank=where(retrieved_docs==most_rel_doc_id)[0] # ID 
        if len(rank)>0:
            in_rank_most_rel_doc_id=1/(rank[0]+1)
    return in_rank_most_rel_doc_id



def MRR(model, judgments, at_K=10):
    '''
    Returns the Mearn Reciprocal Rank of a retrieval model  
    '''
    SRR=0
    for query in tqdm(judgments):
        SRR+=inv_rank_most_relevant_doc(model, query, judgments, at_K) # rank of the most pertinent document
    return SRR/len(judgments) 



def precision(model, at_K=None):
    '''
    
    '''
