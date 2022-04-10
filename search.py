from preprocess import inverted_index_preprocessing, positional_index_preprocessing, simple_preprocessing
from indices import InvertedIndex, PositionalIndex, SubInvertedIndex
from numpy import log10, sqrt, array, mean, linalg, average, arange, argmin, isin, sum as np_sum, zeros, diag
from preprocess import query_correction
from IPython.display import clear_output
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import psutil
import sys

from model_utils import positional_intersect, intersect_increasing_freq, union_many_postings

from search_models import Boolean, Vsm, Bim, BimExt, QueryLklhd, W2Vsm, Lsi

def searching(modl):
    # models={"bool": bool_rtrvl, 'phrase': phrase_rtrvl, "tf_idf": vsm, "bim":bim, 'bim_ext': bim_ext, 
    #         "query_lklh":query_lklh, "w2v_vsm": w2v_vsm, "lsi": lsi}
    models=[Boolean, Vsm, Bim, BimExt, QueryLklhd, W2Vsm, Lsi]

    if type(modl) not in models:
        raise ValueError ("The only supported values for the argument model are of type {}".format(list(models)))

    # Thanks to Gael Guibon (https://gitlab.com/gguibon) for this tip :)
    query = input('Welcome to the best search engine ever :)\nEnter some keywords or type "exit" to stop the engine:')
    while query != "exit":
        results=modl.retrieval(query=query)
        print("Your query: ", query)
        print("\nResult(s): ")
        if results==None: 
            print('Quitting the search engine')
            break

        elif len(results[:modl.top])>0:
            for doc in results[:modl.top]:
                print(doc,'\n')
        else:
            print('No result found. Try another query :)')
        query = input('Enter some keywords, type exit to quit')
        clear_output(wait=True)

# def bool_rtrvl(index, query, tfreqs=None, idfs=None, top=None, query_type='OR', correct_query=False, extension=None, k=None, b=None, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, cluster_centers=None, doc_cluster_labels=None, top_center=None,svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     '''
#     Using boolean retrieval to match the query and documents, where the query is a conjunction of the form
#     term_1 AND term_2 AND ...  AND term_n (where the term_i's are the preprocessed and normalized forms of the query elements) 
#     example the query 'running quickly' should be interpreted as: find the documents where the terms 'run' AND 'quick' appear 
#     (after preprocessing the documents).
#     '''
    
#     if type(index)!=InvertedIndex:
#         raise TypeError ("bool_retrieval supports only an InvertedIndex type")
#     if query_type not in {"OR", "AND"}:
#         raise ValueError ("The only supported values for argument query_type are OR or AND")

#     if correct_query:
#         query=query_correction(query)
#     preprocessed_query=simple_preprocessing(query)
#     overlap=set.intersection(*[set(index.index.keys()), set(preprocessed_query)])
    
#     if query_type=='OR':
#         if len(overlap)==0:
#             return []

#     if query_type=='AND':
#         if len(overlap)<len(preprocessed_query) or len(overlap)==0:
#             return []

#     # We fetch the postings of the common terms
#     postings=[index.index[token] for token in overlap]
#     if index.sort_postings:
#         if query_type=="AND":
#             result_posting=intersect_increasing_freq(postings)
#         else:
#             result_posting=union_many_postings(postings) 

#     if not index.sort_postings:
#         if query_type=="AND":
#             result_posting=set.intersection(*postings) 
#         else:
#             result_posting=set.union(*postings)
#     return [index.documents[ID] for ID in result_posting]





# def phrase_rtrvl(index , query, tfreqs=None, idfs=None, top=None, query_type=None , correct_query=False, extension=None, k=None, b=None, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, clustering_model=None, top_center=None, svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     '''
#     Using boolean retrieval to get documents relevant for a phrase query (e.g the query "hot dog" means find all the documents in       
#     which the term "hot dog" appears, the retrieval engine should not return documents in which the terms "hot"  and "dog" appear       
#     only separately).
#     '''
#     if type(index)!=PositionalIndex:
#         raise TypeError ("phrase_rtrvl supports only an PositionalIndex type")
#     if correct_query:
#         query=query_correction(query)
#     preprocessed_query=positional_index_preprocessing(query)
#     overlap=[term for term in preprocessed_query if term in index.index]
#     #if len(overlap)<len(preprocessed_query):
#     #    return('Your query does not match the documents, please try a new one')
    
#     #else:
#     if index.sort_postings: postings=[index.index[term][1:] for term in overlap] 
#     else: postings=[index.index[term][1] for term in overlap]
#     #print(postings)
#     result=positional_intersect(postings, 1, sort=index.sort_postings) 
#     return [index.documents[ID] for ID in result]

    

# def vsm(index, query,  tfreqs, idfs, top=10, query_type=None, correct_query=False, extension=None, k=None, b=None, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, cluster_centers=None, doc_cluster_labels=None, top_center=None, svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     '''
#     Ranking the collection documents with respect to their scaled (by the norm/length of the query) cosine similarity with the query
#     and returning at most the top N relevant documents.
#     '''
#     if type(index)!=InvertedIndex:
#         raise TypeError ("vsm supports only an InvertedIndex type")
#     if correct_query:
#         query=query_correction(query)

#     doc_scores={ID: 0 for ID in index.documents}
#     preprocessed_query=simple_preprocessing(query)
#     query_term_weights={term: preprocessed_query.count(term)*idfs[term] if term in index.index else 0 \
#                         for term in preprocessed_query
#                         }

#     for term in preprocessed_query:
#         if term in index.index:
#             doc_ids=index.index[term]
#             for doc_id in doc_ids:
#                 doc_scores[doc_id]+=(tfreqs[doc_id][term]*idfs[term]) * query_term_weights[term]
#     # Do not need to normalize by the query norm since it does not impact the ranking
#     doc_scores={ID: doc_scores[ID]/linalg.norm([tfreqs[ID][term]*idfs[term] for term in index.raw_freq[ID]],2) 
#                 for ID in doc_scores
#                }

#     top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
#     if len(top_documents)>0:
#         if top_documents[0][1]==0:
#             return []
#     return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]
        
    

# def bim(index, query, tfreqs=None, idfs=None, top=10, query_type=None, correct_query=False, extension=None, k=None, b=None, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, cluster_centers=None, doc_cluster_labels=None, top_center=None, svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     '''
#     Using binary independence model to rank documents without relevance judgement, a relavance judgment
#     being a kind of notation/feeback from users.
#     '''
#     if type(index)!=InvertedIndex:
#         raise TypeError ("bim supports only an InvertedIndex type")
#     if correct_query:
#         query=query_correction(query)
#     doc_scores={ID: 0 for ID in index.documents}
#     preprocessed_query=simple_preprocessing(query)
#     K=len(index.documents)

#     for term in preprocessed_query:
#         # Query terms that do not appear in the collection are not relevant for ranking the documents
#         if term in index.index:
#             doc_ids=index.index[term]
#             for doc_id in doc_ids:
#                 doc_scores[doc_id]+=log10(0.5*K/len(doc_ids))
#     top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
#     if len(top_documents)>0:
#         if top_documents[0][1]==0:
#             return []
#     return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]



# def bim_ext(index, query, tfreqs, idfs=None, top=10, query_type=None, correct_query=False, extension='bm25', k=1.5, b=0.75, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, cluster_centers=None, doc_cluster_labels=None, top_center=None, svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     '''
#     Using binary independence extensions model to rank the documents, accounting for the term frequencies, document
#     lengths. bm25 is known to be the best among the three extensions: bm25, bm11, two poisson
#     '''
#     if type(index )!=InvertedIndex:
#         return ("bim_ext only support an InvertedIndex type")
#     if extension not in ["bm25", "bm11", "two_poisson"]:
#         raise ValueError ("argument extension takes the only values : bm25, bm11, two_poisson")
#     if correct_query:
#         query=query_correction(query)
#     doc_scores={ID: 0 for ID in index.documents}
#     preprocessed_query=simple_preprocessing(query)
#     K=len(index.documents)
#     l_avg=mean([len(tfreqs[ID]) for ID in index.documents])

#     for term in preprocessed_query:
#         if term in index.index:
#             doc_ids=index.index[term]
#             for doc_id in doc_ids:
#                 freq=tfreqs[doc_id][term]
#                 l_doc=len(tfreqs[doc_id])
#                 extensions={"two_poisson":freq*(k+1)/(freq+k), "bm11":freq*(k+1)/(freq+k*l_doc/l_avg),
#                             "bm25":freq*(k+1)/(k*(1-b)+freq+k*l_doc*b/l_avg)}
#                 doc_scores[doc_id]+=extensions[extension]*log10(0.5*K/len(doc_ids))
                
#     top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
#     if len(top_documents)>0:
#         if top_documents[0][1]==0:
#             return []
#     return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]



# def query_lklh(index, query, tfreqs=None, idfs=None, top=10, query_type=None, correct_query=False, extension=None, k=None, b=None, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, cluster_centers=None, doc_cluster_labels=None, top_center=None, svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     '''
#     Ranking the documents using a language model.
#     Warning terms in model should be preprocessed in the same way than in this function.
#     '''
#     if type(index)!=InvertedIndex:
#         raise TypeError ("query_lklh supports only an InvertedIndex type")
#     if correct_query:
#         query=query_correction(query)
#     preprocessed_query=simple_preprocessing(query)
#     doc_scores={ID: 0 for ID in index.documents}

#     for term in preprocessed_query:
#         if term in index.index:
#             doc_ids=index.index[term]
#             for doc_id in doc_ids:
#                 if doc_scores[doc_id]!=0:
#                     doc_scores[doc_id]*=lang_model[doc_id][term]
#                 else:
#                     doc_scores[doc_id]=lang_model[doc_id][term]

#     top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
#     if len(top_documents)>0:
#         if top_documents[0][1]==0:
#             return []
#     return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]



# def w2v_vsm(index, query, tfreqs, idfs, top=10, query_type=None, correct_query=False, extension=None, k=None, b=None, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, cluster_centers=None, doc_cluster_labels=None, top_center=None, svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     '''
#     Using word2vec model to rank the documents given a query:
#     The idea is to train a word2vec model on a large corpora offline (before querying) use the embeddings to
#     represent the documents (by a weighted average of its constitutent embeddings, eg a tf-idf weight
#     '''
#     if type(index)!=InvertedIndex:
#         raise TypeError ("w2v_vsm supports only an InvertedIndex type")
#     if correct_query:
#         query=query_correction(query)
#     preprocessed_query=simple_preprocessing(query)
    

#     # embedding of the query
#     query_term_weights=[preprocessed_query.count(term)*idfs[term] for term in preprocessed_query if term in idfs]
#     if sum(query_term_weights)>0:
#         query_embed=average(a=[word_embeds[term] for term in preprocessed_query if term in word_embeds], axis=0,\
#                         weights=query_term_weights)
#     else: return []
    
#     if not precluster:
#         doc_scores={ID: cosine_similarity([query_embed,doc_embeds[ID]])[0,1] for ID in tqdm(index.documents)}
#     else:
#         if len(doc_embeds)!=len(doc_cluster_labels):
#             raise ValueError ("Mismatched lenghts: The number of document embeddings should be equal to the number document cluster labels")
#         if top_center>len(cluster_centers):
#             raise ValueError (f'The number of cluster {len(cluster_centers)} should be > top_center {top_center}')
#         # rank the centroids/centers by decreasing cosine similarity with query
#         centers_scores={ID: cosine_similarity([query_embed, cluster_centers[ID]])[0,1]
#                        for ID in range(len(cluster_centers))}
#         top_centers=sorted(centers_scores.items(), key=lambda item: item[1], reverse=True)[:top_center]

#         # rank documents belonging to the clusters of the top centers
#         top_centers_labels=[id for id, score in top_centers]
#         doc_top_clusters=arange(1, len(doc_cluster_labels)+1)[isin(doc_cluster_labels,  top_centers_labels)]
#         doc_scores={ID: cosine_similarity([query_embed, doc_embeds[ID]])[0,1]
#                     for ID in doc_top_clusters}

#     top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
#     if len(top_documents)>0:
#         if top_documents[0][1]==0:
#             return []
#     return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]


    

# def lsi(index, query, tfreqs, idfs, top=10, query_type=None, correct_query=False, extension=None, k=None, b=None, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, cluster_centers=None, doc_cluster_labels=None, top_center=None,svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     '''
#     Using Singular Value Decomposition (SVD) of a word-document matrix (words in row and documents in columns)
#     to rank the documents. This model can be memory consuming
#     '''
#     if type(index) not in [InvertedIndex, SubInvertedIndex]:
#         raise TypeError ("lsi supports only an InvertedIndex type")
#     if correct_query:
#         query=query_correction(query)

#     def run_lsi():
#         preprocessed_query=simple_preprocessing(query)
#         query_term_weights={term: preprocessed_query.count(term)*idfs[term]\
#                             for term in preprocessed_query if term in idfs}
#         u,s,v=svd_word_doc_mat
#         # dense representation of the query (= its projection on the latent topic space)
#         proj_query=sum([query_term_weights[term]*u.T[:,wtoi[term]] for term in query_term_weights\
#                         if term in wtoi]) 
#         # print(len(proj_query))
#         try:
#             if proj_query==0:
#                 return []
#         except:
#             if sum(proj_query)==0:
#                 return []
#         # itod={v:k for k,v in dtoi.items()}
#         # print(dtoi[list(dtoi)[0]])
#         # print(v.shape)

#         # print(s@v[:,dtoi[list(dtoi)[0]]].shape)
#         proj_docs ={ID: (diag(s)@v[:,dtoi[ID]]).reshape(s.shape[0]) for ID in dtoi}
#         # print(proj_docs[list(dtoi)[0]])
#         doc_scores={ID: cosine_similarity([proj_query, proj_docs[ID]])[0,1] for ID in dtoi}

#         top_documents= sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top]
#         if len(top_documents)>0:
#             if top_documents[0][1]==0:
#                 return []
#         return [(index.documents[ID], "score = "+str(score)) for ID, score in top_documents]

#     # check if the model could be run
#     free_memory=psutil.virtual_memory().free  # free memory in bytes
#     int_mem=sys.getsizeof(10) # small integer memory usage in bytes
#     THRESHOLD = 1024 * 1024 * 1024  # 1GB
#     is_able_to_run = (free_memory - THRESHOLD > len(index.index)*len(index.documents)*int_mem)
#     if not is_able_to_run:
#         answer=input("The lsi model may fail due to memory issues. Do you want to pursue ? yes[y] or no[n] ?")
#         n=1
#         while answer not in ("yes", "y", "no", "n") and n<4:
#             answer=input("Do you want to pursue ? yes[y] or no[n] ?")
#             n+=1
#         if answer in ('yes', 'y'): 
#             run_lsi()
#         else: 
#             return 
#     else: run_lsi()
    
            




# def searching(index, modl='tf_idf', tfreqs=None, idfs=None, top=5, query_type=None, correct_query=False, extension=None, k=None, b=None, lang_model=None, smoothing=None, doc_embeds=None, word_embeds=None, precluster=False, cluster_centers=None, doc_cluster_labels=None, top_center=None,svd_word_doc_mat=None, wtoi=None, dtoi=None):
#     models={"bool": bool_rtrvl, 'phrase': phrase_rtrvl, "tf_idf": vsm, "bim":bim, 'bim_ext': bim_ext, 
#             "query_lklh":query_lklh, "w2v_vsm": w2v_vsm, "lsi": lsi}

#     if modl not in models:
#         raise ValueError ("The only supported values for the argument model are {}".format(list(models)))

#     # Thanks to Gael Guibon (https://gitlab.com/gguibon) for this tip :)
#     query = input('Welcome to the best search engine ever :)\nEnter some keywords or type "exit" to stop the engine:')
#     while query != "exit":
#         results=models[modl](index, query, tfreqs, idfs, top, query_type, correct_query, extension, k, b, lang_model, smoothing, doc_embeds, word_embeds, precluster, cluster_centers, doc_cluster_labels, top_center, svd_word_doc_mat, wtoi, dtoi)
#         print("Your query: ", query)
#         print("\nResult(s): ")
#         if results==None: 
#             print('No result found. Try another query :)')
#             query = input('Enter some keywords, type exit to quit')
#             pass

#         elif len(results[:top])>0:
#             for doc in results[:top]:
#                 print(doc,'\n')
#         else:
#             print('No result found. Try another query :)')
#         query = input('Enter some keywords, type exit to quit')
#         clear_output(wait=True)




    
    
    


    
    
    

