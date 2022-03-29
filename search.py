from preprocess import inverted_index_preprocessing, positional_index_preprocessing
from indices import InvertedIndex, PositionalIndex
from numpy import log10, sqrt, array, mean
from preprocess import query_correction



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
    #print(result)
    return result



def positional_intersect(P, k, sort):
    '''
    Intersecting positional posting lists, returning the IDs of the relevant documents
    '''
    result=P[0]
    #print(result)
    rest=P[1:]
    #print(rest)
    while len(rest)>0 and len(result)>0:
        result=positional_intersect_two(result, rest[0], k, sort)
        rest=rest[1:]
    if sort: return [list(dico.keys())[0] for dico in result] 
    return result.keys()



def doc_freq(index, formula='idf'):
    '''
    Computing the document frequency of the index terms
    '''
    terms=index.index.keys()
    if type(index)!=PositionalIndex:
        return("You must use a positional index")
    if formula=='idf':
        return {term: log10(len(index.documents)/index.index[term][0]) if term in index.index else 0 for term in terms}
    elif formula=='prob_idf':
        return {term: log10((len(index.documents)-index.index[term][0])/index.index[term][0]) if term in index.index else 0 for term in terms}
    
    

def term_freq(index, formula='logarithm'):
    '''
    Computing the term frequencies
    '''
    terms=index.index.keys()
    if type(index)!=PositionalIndex:
        return("You must use a positional index")
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
        
        
        
def unigram_model(term_freqs_raw):
    '''
    Computing the unigram language model 
    '''
    return {ID: {term: term_freqs_raw[ID][term]/sum(term_freqs_raw[ID].values()) for term in term_freqs_raw[ID]} for ID in term_freqs_raw}
    
        
        
        
        
        
    
    
def bool_retrieval(index, query, query_type='OR', correct_query=False):
    '''
    Using boolean retrieval to match the query and documents, where the query is a conjunction of the form
    term_1 AND term_2 AND ...  AND term_n (where the term_i's are the preprocessed and normalized forms of the query elements) 
    example the query 'running quickly' should be interpreted as: find the documents where the terms 'run' AND 'quick' appear 
    (after preprocessing the documents).
    '''
    if correct_query:
        query=query_correction(query)
    preprocessed_query=inverted_index_preprocessing(query)
    overlap=set.intersection(*[set(index.index.keys()), preprocessed_query])
    if type(index)!=InvertedIndex:
        return ('The index type should be an inverted index')
    if query_type=='OR':
        print('Retrieving document(s) including at least one of the query terms (in their preprocessed form) ...')
        if len(overlap)==0:
            print('Your query does not match the documents, please try a new one')
            return []
    elif query_type=='AND':
        print('Retrieving document(s) including all the query terms (in their preprocessed form) ...')
        if len(overlap)<len(preprocessed_query):
            print('Your query does not match the documents, please try a new one')
            return []
    # We fetch the postings of the common terms
    postings=[index.index[token] for token in overlap]
    if index.sort_postings:
        result_posting=intersect_increasing_freq(postings) 

    else:
        result_posting=set.intersection(*postings) 
    return [index.documents[ID] for ID in result_posting]



def phrase_retrieval(index , query, correct_query=False):
    '''
    Using boolean retrieval to get documents relevant for a phrase query (e.g the query "hot dog" means find all the documents in       which the term "hot dog" appears, the retrieval engine should not return documents in which the terms "hot"  and "dog" appear       only separately).
    '''
    if type(index)!=PositionalIndex:
        return ("The index should be a positional index")
    if correct_query:
        query=query_correction(query)
    preprocessed_query=positional_index_preprocessing(query)
    overlap=[term for term in preprocessed_query if term in index.index]
    #if len(overlap)<len(preprocessed_query):
    #    return('Your query does not match the documents, please try a new one')
    
    #else:
    if index.sort_postings: postings=[index.index[term][1:] for term in overlap] 
    else: postings=[index.index[term][1] for term in overlap]
    #print(postings)
    result=positional_intersect(postings, 1, sort=index.sort_postings) 
    return [index.documents[ID] for ID in result]

    

def vsm(query, index, term_freqs, doc_freqs, top=10, correct_query=False):
    '''
    Ranking the collection documents with respect to their scaled (by the norm/length of the query) cosine similarity with the query
    and returning at most the top N relevant documents.
    '''
    if type(index)!=PositionalIndex:
        return ("The index should be a positional index")
    
    if correct_query:
        query=query_correction(query)
    doc_scores={ID: 0 for ID in index.documents}
    preprocessed_query=positional_index_preprocessing(query)
    query_term_weights={term: preprocessed_query.count(term)*doc_freqs[term] if term in index.index else 0 \
                        for term in preprocessed_query}
    for term in preprocessed_query:
        if term in index.index:
            if index.sort_postings: doc_ids=[id_ for dico in index.index[term][1:] for id_ in dico] 
            else: doc_ids=index.index[token][1].keys()
            for doc_id in doc_ids:
                doc_scores[doc_id]+=(term_freqs[doc_id][term]*doc_freqs[term]) * query_term_weights[term]
    
    doc_scores={ID: doc_scores[ID]/sqrt(array(list(term_freqs[ID].values()))@array(list(term_freqs[ID].values()))) for ID in doc_scores}
    top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
    if len(top_documents)>0:
        if top_documents[0][1]==0:
            return []
    return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]
        
    

def bim(query, index, top=10, correct_query=False):
    '''
    Using binary independence model to rank documents without relevance judgement
    '''
    if type(index )!=InvertedIndex:
        return ("The index should be an inverted index")
    if correct_query:
        query=query_correction(query)
    doc_scores={ID: 0 for ID in index.documents}
    preprocessed_query=inverted_index_preprocessing(query)
    K=len(index.documents)
    for term in preprocessed_query:
        if term in index.index:
            doc_ids=index.index[term]
            for doc_id in doc_ids:
                doc_scores[doc_id]+=log10(0.5*K/len(doc_ids))
    top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
    if len(top_documents)>0:
        if top_documents[0][1]==0:
            return []
    return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]



def bim_extension(query, index, term_freqs, extension='two_poisson', k=1.5, b=0.75, top=10, correct_query=False):
    '''
    Using binary independence extensions model to rank the documents, accounting for the term frequencies, document
    lengths.
    '''
    if type(index )!=PositionalIndex:
        return ("The index should be a positional index")
    if correct_query:
        query=query_correction(query)
    doc_scores={ID: 0 for ID in index.documents}
    preprocessed_query=positional_index_preprocessing(query)
    K=len(index.documents)
    l_avg=mean([len(term_freqs[ID]) for ID in index.documents])
    for term in preprocessed_query:
        if term in index.index:
            if index.sort_postings:doc_ids=[id_ for dico in index.index[term][1:] for id_ in dico]
            else: doc_ids=index.index[term][1].keys()
            for doc_id in doc_ids:
                freq=term_freqs[doc_id][term]
                l_doc=len(term_freqs[doc_id])
                if extension=='two_poisson':
                    doc_scores[doc_id]+=(freq*(k+1)/(freq+k))*log10(0.5*K/len(doc_ids))
                elif extension=='bm11':
                    doc_scores[doc_id]+=(freq*(k+1)/(freq+k*l_doc/l_avg))*log10(0.5*K/len(doc_ids))
                elif extension=='bm25':
                    doc_scores[doc_id]+=(freq*(k+1)/(k*(1-b)+freq+k*l_doc*b/l_avg))*log10(0.5*K/len(doc_ids))
    top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
    if len(top_documents)>0:
        if top_documents[0][1]==0:
            return []
    return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]



def query_likelihood(query, index, model, top=10, smoothing=None, kind='unigram', correct_query=False):
    '''
    Ranking the documents using a language model.
    Warning terms in model should be preprocessed in the same way than in this function.
    '''
    if type(index )!=InvertedIndex:
        return ("The index should be an inverted index")
    if correct_query:
        query=query_correction(query)
    preprocessed_query=inverted_index_preprocessing(query)
    doc_scores={ID: 0 for ID in index.documents}
    for term in preprocessed_query:
        if term in index.index:
            doc_ids=index.index[term]
            for doc_id in doc_ids:
                if doc_scores[doc_id]!=0:
                    if kind=='unigram': doc_scores[doc_id]*=model[doc_id][term]
                else:
                    doc_scores[doc_id]=model[doc_id][term]
    top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
    if len(top_documents)>0:
        if top_documents[0][1]==0:
            return []
    return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]




    
    
    


    
    
    

