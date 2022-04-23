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
        SRR+=inv_rank_most_relevant_doc(model, query, judgments, at_K) 
    return SRR/len(judgments) 



def precision(model, query, judgments):
    '''
    Returns the precision of a given model for a query.
    It is not precision@K because we do not set a priori the rank K, instead this precision
    depends on the available jugments for the query
    '''
    prec=0
    rel_doc_id=judgments[query].keys()
    model.top=len(judgments[query]) # to retrieve only a number of documents equal to the number of available judgments
    results=model.retrieval(query)
    retrieved_docs=array(results)[:,0] # IDs of the retrieved docs
    if len(retrieved_docs)>0:
        prec=sum([ID in retrieved_docs for ID in rel_doc_id])/len(retrieved_docs)
    return prec


def MAP(model, judgments):
    '''
    Returns the Mean Average Precision (not MAP@K) of a model given jugments
    '''
    SAP=0
    for query in tqdm(judgments):
        SAP+=precision(model, query, judgments) 
    return SAP/len(judgments) 


def run_benchmark(benchmark, judgments):
    '''
    Model(s) evaluation
    '''
    for model_name in benchmark:
        models=benchmark[model_name]["model"]
        at_Ks=list(benchmark[model_name]["MRR@"].keys())
        print(f'Running {len(models)} {model_name} model(s)')
        for pos, model in enumerate(models):
            print(f'\t{pos+1} model over {len(models)}')
            for K in at_Ks:
                benchmark[model_name]["MRR@"][K].append(MRR(model, judgments, at_K=K))
            benchmark[model_name]["MAP"].append(MAP(model, judgments))
    return benchmark
            