from dataclasses import dataclass, field
from indices import InvertedIndex, PositionalIndex, SubInvertedIndex 
from numpy import array
from typing import Union
from preprocess import query_correction, simple_preprocessing
from model_utils import intersect_increasing_freq, union_many_postings, rank_documents
from numpy import linalg, log10, mean, average, arange, isin, diag
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm




@dataclass
class Arguments:
    index: Union[InvertedIndex, PositionalIndex, SubInvertedIndex] = InvertedIndex() 
    top: int=5
    correct_query: bool=False 
    args: dict=field(default_factory=dict)

    # @property
    # def _check_type(self):

    # def _add_args(self, name, value):
    #     '''
    #     Adding some model arguments to args
    #     '''
    #     for pos,n in enumerate(name): 
    #         self.args[n]=value[pos]




@dataclass
class Boolean(Arguments):
    query_type: str="AND"
    name: str='boolean'
    
    def retrieval(self, query):
        '''
        Using boolean retrieval to match the query and documents, where the query is a conjunction of the form
        term_1 AND term_2 AND ...  AND term_n (where the term_i's are the preprocessed and normalized forms of the query elements) 
        example the query 'running quickly' should be interpreted as: find the documents where the terms 'run' AND 'quick' appear 
        (after preprocessing the documents).
        '''
        
        if type(self.index)!=InvertedIndex:
            raise TypeError ("bool_retrieval supports only an InvertedIndex type")
        if self.query_type not in {"OR", "AND"}:
            raise ValueError ("The only supported values for argument query_type are OR or AND")

        if self.correct_query:
            query_correction(query, self.index)
        preprocessed_query=simple_preprocessing(query)
        overlap=set.intersection(*[set(self.index.index.keys()), set(preprocessed_query)])
        
        if self.query_type=='OR':
            if len(overlap)==0:
                return []

        if self.query_type=='AND':
            if len(overlap)<len(preprocessed_query) or len(overlap)==0:
                return []

        # We fetch the postings of the common terms
        postings=[self.index.index[token] for token in overlap]
        if self.index.sort_postings:
            if self.query_type=="AND":
                result_posting=intersect_increasing_freq(postings)
            else:
                result_posting=union_many_postings(postings) 

        if not self.index.sort_postings:
            if self.query_type=="AND":
                result_posting=set.intersection(*postings) 
            else:
                result_posting=set.union(*postings)

        # A slack score equal to 1 is added to mean that the document is retrieved
        # top does not necessarily make sense for boolean retrieval, but is added to
        # avoid printing to much outputs when searching
        return [(ID,1) for ID in result_posting][:self.top]


# PENDING
# @dataclass
# class Phrase(Arguments):


@dataclass
class Vsm(Arguments):
    tfreqs: dict=field(default_factory=dict)
    idfs: dict=field(default_factory=dict)
    name: str='vsm' 
    

    def retrieval(self, query):
        '''
        Ranking the collection documents with respect to their scaled (by the norm/length of the query) cosine similarity with the query
        and returning at most the top N relevant documents.
        '''
        if type(self.index)!=InvertedIndex:
            raise TypeError ("vsm supports only an InvertedIndex type")
        if self.correct_query:
            query_correction(query, self.index)

        doc_scores={ID: 0 for ID in self.index.documents}
        preprocessed_query=simple_preprocessing(query)
        query_term_weights={term: preprocessed_query.count(term)*self.idfs[term] if term in self.index.index else 0 \
                            for term in preprocessed_query
                            }

        for term in preprocessed_query:
            if term in self.index.index:
                doc_ids=self.index.index[term]
                for doc_id in doc_ids:
                    doc_scores[doc_id]+=(self.tfreqs[doc_id][term]*self.idfs[term]) * query_term_weights[term]
        # Do not need to normalize by the query norm since it does not impact the ranking
        doc_scores={ID: doc_scores[ID]/linalg.norm([self.tfreqs[ID][term]*self.idfs[term] for term in self.index.raw_freq[ID]],2) 
                    for ID in doc_scores
                }
        
        return rank_documents(doc_scores, self.top)



@dataclass
class Bim(Arguments):
    name: str='bim'

    def retrieval(self, query):
        '''
        Using binary independence model to rank documents without relevance judgement, a relavance judgment
        being a kind of notation/feeback from users.
        '''
        if type(self.index)!=InvertedIndex:
            raise TypeError ("bim supports only an InvertedIndex type")
        if self.correct_query:
            query_correction(query, self.index)
        doc_scores={ID: 0 for ID in self.index.documents}
        preprocessed_query=simple_preprocessing(query)
        K=len(self.index.documents)

        for term in preprocessed_query:
            # Query terms that do not appear in the collection are not relevant for ranking the documents
            if term in self.index.index:
                doc_ids=self.index.index[term]
                for doc_id in doc_ids:
                    doc_scores[doc_id]+=log10(0.5*K/len(doc_ids))

        return rank_documents(doc_scores, self.top)



@dataclass
class BimExt(Arguments):
    k: Union[float, int]=1.5
    b: Union[float, int]=0.75
    # extension: str="bm25"
    name: str='bim_extension'

    def retrieval(self, query):
        '''
        Using binary independence extensions model to rank the documents, accounting for the term frequencies, document
        lengths. bm25 is known to be the best among the three extensions: bm25 (when k!=0 and b!=0), bm11 (when b=1 and k!=0), 
        two poisson (when b=0 and k!=0)
        '''
        if type(self.index)!=InvertedIndex:
            raise TypeError ("bim_ext only support an InvertedIndex type")
        # if self.extension not in ["bm25", "bm11", "two_poisson"]:
        #     raise ValueError ("argument extension takes the only values : bm25, bm11, two_poisson")
        if self.correct_query:
            query_correction(query, self.index)
        
        self.name='bim_extension'+'_k='+str(self.k)+'_b='+str(self.b)

        doc_scores={ID: 0 for ID in self.index.documents}
        preprocessed_query=simple_preprocessing(query)
        K=len(self.index.documents)
        l_avg=mean([len(self.index.raw_freq[ID]) for ID in self.index.documents])

        for term in preprocessed_query:
            if term in self.index.index:
                doc_ids=self.index.index[term]
                for doc_id in doc_ids:
                    freq=self.index.raw_freq[doc_id][term]
                    l_doc=len(self.index.raw_freq[doc_id])
                    adjustment=freq*(self.k+1)/(self.k*(1-self.b)+freq+self.k*l_doc*self.b/l_avg)
                    doc_scores[doc_id]+=adjustment*log10(0.5*K/len(doc_ids))
                    
        return rank_documents(doc_scores, self.top)



@dataclass
class QueryLklhd(Arguments):
    lang_model: dict=field(default_factory=dict)
    name: str='query_likelihood'

    def retrieval(self, query):
        '''
        Ranking the documents using a language model.
        Warning terms in model should be preprocessed in the same way than in this function.
        '''
        if type(self.index)!=InvertedIndex:
            raise TypeError ("query_lklhd supports only an InvertedIndex type")
        if self.correct_query:
            query_correction(query, self.index)

        preprocessed_query=simple_preprocessing(query)
        doc_scores={ID: 0 for ID in self.index.documents}

        for term in preprocessed_query:
            if term in self.index.index:
                doc_ids=self.index.index[term]
                for doc_id in doc_ids:
                    if doc_scores[doc_id]!=0:
                        doc_scores[doc_id]*=self.lang_model[doc_id][term]
                    else:
                        doc_scores[doc_id]=self.lang_model[doc_id][term]

        return rank_documents(doc_scores, self.top)



@dataclass
class W2Vsm(Arguments):
    word_embeds: dict=field(default_factory=dict)
    idfs: dict=field(default_factory=dict) 
    precluster: bool=False
    cluster_centers: dict=field(default_factory=dict)
    doc_cluster_labels: Union[array, list]=array([])
    top_center: int=50
    doc_embeds: dict=field(default_factory=dict)
    name: str='w2vsm'

    def retrieval(self, query):
        '''
        Using word2vec model to rank the documents given a query:
        The idea is to train a word2vec model on a large corpora offline (before querying) use the embeddings to
        represent the documents (by a weighted average of its constitutent embeddings, eg a tf-idf weight
        '''
        if type(self.index)!=InvertedIndex:
            raise TypeError ("w2vsm supports only an InvertedIndex type")
        if self.correct_query:
            query_correction(query, self.index)
        preprocessed_query=simple_preprocessing(query)
        

        # embedding of the query, a term in the query should appear both in the word embeddings and the idfs to be taken into account
        query_term_weights=[preprocessed_query.count(term)*self.idfs[term] for term in preprocessed_query\
                            if term in self.word_embeds and term in self.idfs]
        if sum(query_term_weights)>0:
            query_embed=average(a=[self.word_embeds[term] for term in preprocessed_query 
                        if term in self.word_embeds and term in self.idfs], axis=0,\
                            weights=query_term_weights)
        else: return []
        
        if not self.precluster:
            doc_scores={ID: cosine_similarity([query_embed,self.doc_embeds[ID]])[0,1] for ID in tqdm(self.index.documents)}
        else:
            if len(self.doc_embeds)!=len(self.doc_cluster_labels):
                raise ValueError ("Mismatched lenghts: The number of document embeddings should be equal to the number document cluster labels")
            if self.top_center>len(self.cluster_centers):
                raise ValueError (f'The number of cluster {len(self.cluster_centers)} should be > top_center {self.top_center}')

            # rank the centroids/centers by decreasing cosine similarity with query
            centers_scores={ID: cosine_similarity([query_embed, self.cluster_centers[ID]])[0,1]
                        for ID in range(len(self.cluster_centers))}
            top_centers=sorted(centers_scores.items(), key=lambda item: item[1], reverse=True)[:self.top_center]

            # rank documents belonging to the clusters of the top centers
            top_centers_labels=[id for id, score in top_centers]
            doc_top_clusters=arange(1, len(self.doc_cluster_labels)+1)[isin(self.doc_cluster_labels,  top_centers_labels)]
            doc_scores={ID: cosine_similarity([query_embed, self.doc_embeds[ID]])[0,1]
                        for ID in doc_top_clusters}

        return rank_documents(doc_scores, self.top)



@dataclass
class Lsi(Arguments):
    svd_word_doc_mat: Union[array, list]=array([])
    wtoi: dict=field(default_factory=dict)
    dtoi: dict=field(default_factory=dict)
    idfs: dict=field(default_factory=dict)
    name: str='lsi'

    def retrieval(self, query):
        '''
        Using Singular Value Decomposition (SVD) of a word-document matrix (words in row and documents in columns)
        to rank the documents. This model can be memory consuming
        '''
        if type(self.index) not in [InvertedIndex, SubInvertedIndex]:
            raise TypeError ("lsi supports only an InvertedIndex type")
        if self.correct_query:
            query=query_correction(query)

        preprocessed_query=simple_preprocessing(query)
        query_term_weights={term: preprocessed_query.count(term)*self.idfs[term]\
                            for term in preprocessed_query if term in self.idfs}
        u,s,v=self.svd_word_doc_mat

        # dense representation of the query (= its projection on the latent topic space)
        proj_query=sum([query_term_weights[term]*u.T[:, self.wtoi[term]] for term in query_term_weights\
                        if term in self.wtoi]) 
        try:
            if proj_query==0:
                return []
        except:
            if sum(proj_query)==0:
                return []

        proj_docs ={ID: (diag(s)@v[:,self.dtoi[ID]]).reshape(s.shape[0]) for ID in self.dtoi}
        doc_scores={ID: cosine_similarity([proj_query, proj_docs[ID]])[0,1] for ID in self.dtoi}

        return rank_documents(doc_scores, self.top)


# PENDING
# @dataclass
# class MatchPyramid(Arguments):

