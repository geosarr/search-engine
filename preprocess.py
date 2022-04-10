from nltk.corpus import stopwords, words
import string
from nltk.stem import PorterStemmer
import re
import numpy as np

import nltk
nltk.download("words")
nltk.download('stopwords')

from termcolor import colored

def clean(text):
    # punctuations= ''.join(set(string.punctuation)-{'-'})      
    punctuations= ''.join(set(string.punctuation))         
    preproc_text=text.lower().replace('\n', " ").replace("\t", " ").translate(str.maketrans(' ', ' ', punctuations))


    # dropping multiple whitespaces
    preproc_text=re.sub(' +', ' ', preproc_text)
        
    return preproc_text



def levenshtein_distance(char1,char2):
    '''
    Computing the Levenshtein Distance between two string characters char1 and char2
    '''
    char1=char1.strip()
    char2=char2.strip()
    n1,n2=len(char1),len(char2)
    m=np.zeros((n1,n2))
    for i in range(n1):
        m[i,0]=i
    for j in range(n2):
        m[0,j]=j
    for i in range(1, n1):
        for j in range(1, n2):
            if char1[i]==char2[j]:
                m[i,j]=min(m[i-1,j]+1, m[i,j-1]+1, m[i-1,j-1])
            else: m[i,j]=min(m[i-1,j]+1, m[i,j-1]+1, m[i-1,j-1]+1)
    return m[n1-1,n2-1]



def character_ngram(word,n=2):
    '''
    Returning the character n-grams of word
    '''
    w=word.strip()
    if n>=len(w):
        return {w}
    return {w[i:i+n] for i in range(len(w)-n+1)}



def query_correction(query, index, threshold=0.7, dictionary=set(words.words("en"))):
    '''
    Correcting the query using character n-grams to prefilter the dictionary candidates
    '''
    cleaned_query_terms=clean(query.strip()).split()
    # Flag for correction
    correction=False
    # The different character grams in the collection
    chars=set(index.char_t_index.keys()) 
    for position, term in enumerate(cleaned_query_terms):
        # Fetching the collection terms that contain at least one of the character n-gram of the current term
        
        # The character gram of the current term
        term_chars=character_ngram(term, index.ngram) 
        #print(term_chars)
        overlap_chars=set.intersection(*[term_chars, chars])
        #print(overlap_chars)
        if len(overlap_chars)>0:
            overlap_terms=set.union(*[index.char_t_index[t] for t in overlap_chars]) 
            #print(overlap_terms)
            # Getting the 100*threshold % terms that share a higher number of character grams with the current term
            jaccard_coefs={t: len(set.intersection(*[index.t_char_index[t], term_chars]))/\
                                 len(set.union(*[index.t_char_index[t], term_chars])) \
                           for t in  overlap_terms
                           }
            #print(jaccard_coefs)
            sorted_jaccard_coefs=sorted(jaccard_coefs.items(), key=lambda item: item[1], reverse=True)
            #print(sorted_jaccard_coefs)
            top_terms=set(dict(sorted_jaccard_coefs[:int(threshold*len(jaccard_coefs))+1]).keys())
            # print(top_terms)
            # Getting the closest dictionary term 
            overlap_dict_top_terms=top_terms.intersection(dictionary)
            #print(overlap_dict_top_terms)
            distance=np.inf
            corrected_term=term
            for t in overlap_dict_top_terms:
                if levenshtein_distance(t, term)<distance:
                    corrected_term,distance=t,levenshtein_distance(t, term)
            cleaned_query_terms[position]=corrected_term
            if corrected_term!=term:
                correction=True
        #else: unknown.append(term)
        
    corrected_query=" ".join(cleaned_query_terms)
    if correction:
        print('Did you mean: %s ?' % (colored(corrected_query, "cyan")))
        print('\n')
                        
            
def simple_preprocessing(text):
    '''
    Simpler version of preprocessing
    with this version indexing is much faster, i.e the stemming part is time consuming -> Is it worth doing it ?
    It depends on the result of the search engine.
    '''
    return clean(text).split()


def inverted_index_preprocessing(text):
    '''
    Preprocessing the documents for inverted indexing purpose.
    '''
    def tokenization(text):
        '''
        Breaking down the texts into instances (tokens) using white spaces after lower casing 
        the elements of the text and dropping the punctuations 
        '''
        preproc_text=clean(text)
        
        return set(preproc_text.split())
    
    def stop_words_removal(tokens):
        '''
        Removing the stop words (some elements of the text that do not bear significant 
        meaning and may lead to false positives), using a lexicon of stop words.
        tokens should be a set
        '''
        return tokens - set(stopwords.words('english'))
    
    def stemming(tokens):
        ''' 
        Standardizing the words using a stemming method that will transform
        the word into their grammatical root, Porter algorithm is used
        '''
        return {PorterStemmer().stem(token) for token in tokens}
    
    return stemming(stop_words_removal(tokenization(text)))
                


def positional_index_preprocessing(text):
    '''
    Preprocessing the text for positional indexing purpose
    '''
    def tokenization(text):
        preproc_text=clean(text)
        
        return preproc_text.split()
    
    def stop_words_removal(tokens):
        return [token for token in tokens if token not in stopwords.words('english')]
    
    def stemming(tokens):
        return [PorterStemmer().stem(token) for token in tokens]
    
    return stemming(stop_words_removal(tokenization(text)))
