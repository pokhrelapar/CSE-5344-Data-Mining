"""
    Dependencies:
        Python version 3.12.3, 
        nltk version 3.9.1
    
    Setup:
        conda install/ pip install packages
        ensure  nltk.corpus.stopwords is downloaded
    
"""

import os
import math
import string
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer


ROOT_FOLDER = "./US_Inaugural_Addresses"

tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
stemmer = PorterStemmer()

nltk.download("stopwords")

stop_words = stopwords.words("english")
stop_words = set(stop_words)


def read_process_files(root_folder):
    """
    Processes all the .txt files

    """
    speech_documents = {}

    for filename in os.listdir(root_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(root_folder, filename), "r") as file_path:
                data = file_path.read().lower()
                speech_documents[filename] = data

    return speech_documents


def remove_stopwords(tokens):
    """
    remove stop words from a text document

    """
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


def stem_tokens(tokens):
    """
    stem words using PorterStemmer

    """
    return [stemmer.stem(word) for word in tokens]


def tokenize_and_process(text):
    """
    tokenize, remove stop words and stem words

    """
    tokens = tokenizer.tokenize(text)  # Tokenize
    filtered_tokens = remove_stopwords(tokens)  # Remove stopwords
    stemmed_tokens = stem_tokens(filtered_tokens)  # Perform stemming
    return stemmed_tokens


def calculate_term_frequency(tokens):
    """
    find tf of a token

    """
    tf = defaultdict(int)

    for token in tokens:
        tf[token] += 1
    return tf


def calculate_document_frequency(documents):
    """
    find df of a token across documents

    """

    df = defaultdict(int)

    for tokens in documents.values():
        new_tokens = set(tokens)
        for token in new_tokens:
            df[token] += 1
    return df


def normalize_vector(vector):
    """
    normalize operations

    """
    norm = math.sqrt(sum(value**2 for value in vector.values()))

    normalized_vector = (
        {key: value / norm for key, value in vector.items()} if norm > 0 else vector
    )

    return normalized_vector


def calculate_tf_idf(documents, term_freq, dcoument_freq, total_documents):
    """
    find tf-idf weight of a token

    """

    # stores tf-idf vector for each document
    tf_idf_vectors = {}

    # go through each tokens in a document
    for document, tokens in documents.items():
        tf_idf_vector = defaultdict(float)
        tf = term_freq[document]

        # go through each word in tf dictionary
        for token in tf:
            tf_value = 1 + math.log10(tf[token]) if tf[token] > 0 else 0
            idf_value = (
                math.log10(total_documents / dcoument_freq[token])
                if dcoument_freq[token] > 0
                else 0
            )

            tf_idf_value = tf_value * idf_value
            tf_idf_vector[token] = tf_idf_value

        # normalize to make  tf_idf vector with unit legth
        tf_idf_vectors[document] = normalize_vector(tf_idf_vector)

    return tf_idf_vectors


speech_documents = read_process_files(ROOT_FOLDER)


# show original documents and inagural speech
"""
for i, e in enumerate(speech_documents.items()):
  print(i, e)
"""

processed_documents = {}

for document, text in speech_documents.items():
    processed_documents[document] = tokenize_and_process(text)


# show processed documents after tokenizing, stopword removal and stemming
"""
for i, e in enumerate(processed_documents.items()):
  print(i, e)
"""

term_frequencies = {}

for document, tokens in processed_documents.items():
    term_frequencies[document] = calculate_term_frequency(tokens)


document_frequencies = calculate_document_frequency(processed_documents)


N = len(processed_documents)


tf_idf_vectors = calculate_tf_idf(
    processed_documents, term_frequencies, document_frequencies, N
)


def calculate_query_vector(query, document_frequencies, N):
    """
    calculate the normalized query vector

    """

    query_tokens = tokenize_and_process(query)
    query_tf = calculate_term_frequency(query_tokens)

    query_vector = defaultdict(float)

    for token in query_tf:
        tf_value = 1 + math.log10(query_tf[token]) if query_tf[token] > 0 else 0
        idf_value = (
            math.log10(N / document_frequencies[token])
            if document_frequencies[token] > 0
            else 0
        )
        query_vector[token] = tf_value * idf_value

    query_vector = normalize_vector(query_vector)

    return query_vector


def create_postings_list(td_idf_vectors):
    """
    create posting list of the tf-idf vectors

    """
    postings_list = defaultdict(list)

    for document, vector in td_idf_vectors.items():
        for token, weight in vector.items():
            postings_list[token].append((document, weight))

    for token in postings_list:
        postings_list[token].sort(key=lambda x: x[1], reverse=True)

    return postings_list


def show_top_10_postings(query_tokens, postings_list):
    """
    return the top 10 posting list of the tf-idf vectors

    """
    top_10_postings = defaultdict(list)
    for token in query_tokens:
        if token in postings_list:
            top_10_postings[token] = postings_list[token][:10]

    return top_10_postings


def getidf(token):
    """
    get idf value of a token

    """
    stemmed_token = stemmer.stem(token)

    if stemmed_token not in document_frequencies:
        return -1

    df = document_frequencies.get(stemmed_token, 0)

    if df > 0:
        return math.log10(N / df)

    return -1


def getweight(filename, token):
    """
    get tf-idf weight of a token in a document

    """

    stemmed_token = stemmer.stem(token)
    tf_idf_weight = tf_idf_vectors.get(filename, {}).get(stemmed_token, 0)

    return tf_idf_weight


def query(qstring):
    """
    find the best matching document for the given query

    """

    query_vector = calculate_query_vector(qstring, document_frequencies, N)
    query_tokens = set(query_vector.keys())

    postings_list = create_postings_list(tf_idf_vectors)

    # if no document contains any token in the query
    if not any(token in postings_list for token in query_tokens):
        return ("None", 0)

    # top-10 entries for each query token
    top_10_per_token = show_top_10_postings(query_tokens, postings_list)

    candidate_documents = defaultdict(float)  #  actual similarity scores
    upper_bound_documents = defaultdict(float)  #  upper-bound scores

    docs_token_appear = defaultdict(set)

    min_upper_bound = float("-inf")

    for token, w_tq in query_vector.items():
        if token not in top_10_per_token:
            continue

        top_10_list = top_10_per_token[token]
        tenth_weight = top_10_list[-1][1] if len(top_10_list) == 10 else 0

        top_10_doc_ids = {doc_id for doc_id, _ in top_10_list}

        for doc_id, w_td in top_10_list:
            candidate_documents[doc_id] += w_tq * w_td
            docs_token_appear[doc_id].add(token)

        for doc_id in candidate_documents:
            if doc_id not in top_10_doc_ids:  # If not in top-10, add upper bound
                upper_bound_documents[doc_id] += w_tq * tenth_weight

    best_doc = None
    best_actual_score = float("-inf")

    for doc_id, actual_score in candidate_documents.items():
        # if len(docs_token_appear[doc_id]) < len(query_tokens):
        #     continue

        upper_bound_score = actual_score + upper_bound_documents.get(doc_id, 0)
        min_upper_bound = max(min_upper_bound, upper_bound_score)

        if actual_score >= min_upper_bound:
            best_doc = doc_id
            best_actual_score = actual_score

    # if need more than top 10 elements
    if best_doc is None:
        return ("fetch more", 0)

    return (best_doc, best_actual_score)


print("---------------------")
print("|       Results     |")
print("---------------------")

print("\n -------IDF-------")
print("%.12f" % getidf("british"))
print("%.12f" % getidf("union"))
print("%.12f" % getidf("dollar"))
print("%.12f" % getidf("constitution"))
print("%.12f" % getidf("power"))


print("\n -------Weight-------")
print("%.12f" % getweight("19_lincoln_1861.txt", "states"))
print("%.12f" % getweight("07_madison_1813.txt", "war"))
print("%.12f" % getweight("05_jefferson_1805.txt", "false"))
print("%.12f" % getweight("22_grant_1873.txt", "proposition"))
print("%.12f" % getweight("16_taylor_1849.txt", "duties"))


print("\n ----- QUERY ---------")
print("(%s, %.12f)" % query("executive power"))
print("(%s, %.12f)" % query("foreign government"))
print("(%s, %.12f)" % query("public rights"))
print("(%s, %.12f)" % query("people government"))
print("(%s, %.12f)" % query("states laws"))
