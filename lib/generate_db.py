import sys
import math
import sqlite3
import numpy as np
from numpy import argsort, asarray

### utility functions ###

def generic_generator(*args):
    """
    A helpful abstraction to pass results to sqlite3.executemany
    """
    for zips in zip(*args):
        yield zips

def file_generator(fhandle):
    """
    A generator for an open file handle that strips lines
    """
    for line in fhandle:
        yield line.strip()

### score functions ###

def get_doc_score(doca, docb, axis=1):
    """
    Returns 1/2 times the sum of squares distance between docuemnts.

    doca is expected to be a 1d array while docb can be 2d.
    """
    return .5 * np.sum((doca - docb)**2, axis=axis)

def get_topic_score(topica, topicb, axis=1):
    """
    Returns discrete Hellinger distance between topics

    topica is expected to be a 1d array, while topicb can be 2d.
    """
    score = np.sum((np.abs(topica)**.5 - np.abs(topicb)**.5)**2, axis=axis)
    return 0.5 * score / (100. * len(topica))

def get_term_score(terma, termb, axis=1):
    """
    Returns sum of squares distance of term pairs

    terma is expected to be a 1d array, while termb can be 2d.
    """
    return np.sum((terma - termb)**2, axis=axis)

### write relations to db functions ###

def write_doc_doc(con, cur, gamma_file):
    cur.execute('CREATE TABLE doc_doc (id INTEGER PRIMARY KEY, doc_a INTEGER, '
                'doc_b INTEGER, score FLOAT)')
    cur.execute('CREATE INDEX doc_doc_idx1 ON doc_doc(doc_a)')
    cur.execute('CREATE INDEX doc_doc_idx2 ON doc_doc(doc_b)')
    con.commit()

    docs = np.loadtxt(gamma_file) ** 2

    # get the closest 100 relations per document
    # NOTE: this is buggy, consider this case
    # you can't have multiple keys with the same value even if they exist
    for a in range(len(docs)):
        doc = docs[a]
        # index below by a, because already compared before a
        distance = get_doc_score(doc, docs[a:])
        # drop zeros
        distance[distance == 0] = np.inf
        min_doc_idx = np.argsort(distance)[:100]

        # generator of many results
        res = generic_generator((str(a),)*100, map(str, min_doc_idx),
                distance[min_doc_idx])

        execution_string = 'INSERT INTO doc_doc (id, doc_a, doc_b, score) '
        execution_string += 'VALUES(NULL, ?, ?, ?)'

        cur.executemany(execution_string, res)

    con.commit()

def write_doc_topic(con, cur, gamma_file):
    cur.execute('CREATE TABLE doc_topic (id INTEGER PRIMARY KEY, doc INTEGER, '
                'topic INTEGER, score FLOAT)')
    cur.execute('CREATE INDEX doc_topic_idx1 ON doc_topic(doc)')
    cur.execute('CREATE INDEX doc_topic_idx2 ON doc_topic(topic)')
    con.commit()

    docs = np.loadtxt(gamma_file, 'r')
    # for each line in the gamma file
    for doc_no,doc in enumerate(open(gamma_file, 'r')):
        doc = map(float, doc.split())
        ins = 'INSERT INTO doc_topic (id, doc, topic, score) '
        ins += 'VALUES(NULL, ?, ?, ?)'
        res = generic_generator((doc_no,)*len(doc), range(len(doc)), doc)
        cur.executemany(ins, res)

    con.commit()

def write_topics(con, cur, beta_file, vocab):
    """
    For each topic, write the first 3 most probably words to
    """
    cur.execute('CREATE TABLE topics (id INTEGER PRIMARY KEY, title VARCHAR(100))')
    con.commit()

    #NOTE: What is the following line for and why doesn't it raise an error?
    topics_file = open(filename, 'a')

    for topic in open('final.beta', 'r'):
        topic = map(float, topic.split())
        index = argsort(topic)
        ins = 'INSERT INTO topics (id, title) VALUES(NULL, ?)'
        buf = "{%s, %s, %s}" % (vocab[index[0]],
                                vocab[index[1]],
                                vocab[index[2]])
        cur.execute(ins, [buffer(buf)])

    con.commit()

def write_topic_term(con, cur, beta_file):
    cur.execute('CREATE TABLE topic_term (id INTEGER PRIMARY KEY, topic INTEGER, '
                'term INTEGER, score FLOAT)')
    cur.execute('CREATE INDEX topic_term_idx1 ON topic_term(topic)')
    cur.execute('CREATE INDEX topic_term_idx2 ON topic_term(term)')
    con.commit()

    topic_term_file = open(filename, 'a')

    for topic_no,topic in enumerate(open(beta_file, 'r')):
        topic = asarray(topic.split(), dtype=float)
        index = argsort(topic)
        res = generic_generator((topic_no,) * len(topic),
                                index, topic[index])
        ins = 'INSERT INTO topic_term (id, topic, term, score) '
        ins += 'VALUES(NULL, ?, ?, ?)'
        cur.executemany(ins, res)

    con.commit()

def write_topic_topic(con, cur, beta_file):
    cur.execute('CREATE TABLE topic_topic (id INTEGER PRIMARY KEY, '
                'topic_a INTEGER, topic_b INTEGER, score FLOAT)')
    cur.execute('CREATE INDEX topic_topic_idx1 ON topic_topic(topic_a)')
    cur.execute('CREATE INDEX topic_topic_idx2 ON topic_topic(topic_b)')
    con.commit()

    # for each line in the beta file
    read_file = open(beta_file, 'r')
    topics = []
    for topic in read_file:
        topics.append(map(float, topic.split()))
    topics = np.asarray(topics)

    for topica_count,topic in enumerate(topics):
        #index by count because distance is symmetric
        scores = get_topic_score(topic, topics[topica_count:])
        res = generic_generator((topica_count,)*len(scores),
                                range(len(scores)),
                                scores)
        ins = 'INSERT INTO topic_topic (id, topic_a, topic_b, score) '
        ins += 'VALUES(NULL, ?, ?, ?)'
        con.executemany(ins, res)
    con.commit()

def write_term_term(con, cur, beta_file, no_vocab):
    cur.execute('CREATE TABLE term_term (id INTEGER PRIMARY KEY, '
                'term_a INTEGER, term_b INTEGER, score FLOAT)')
    cur.execute('CREATE INDEX term_term_idx1 ON term_term(term_a)')
    cur.execute('CREATE INDEX term_term_idx2 ON term_term(term_b)')
    con.commit()

    v = []
    for topic in file(beta_file, 'r'):
        v.append(map(float, topic.split()))
    v = np.exp(v)**.5

    for a in range(len(v)):
        terma = v[a]
        score = get_term_score(terma, v[a:])
        # drop zeros
        score[score == 0] = np.inf
        min_score_idx = np.argsort(score)[:100]
        res = generic_generator((str(a),)*len(score),
                                map(str, min_score_idx),
                                score[min_score_idx])
        ins = 'INSERT INTO term_term (id, term_a, term_b, score) '
        ins += 'VALUES(NULL, ?, ?, ?)'
        cur.executemany(ins, res)

    con.commit()

def write_doc_term(con, cur, wordcount_file, no_words):
    cur.execute('CREATE TABLE doc_term (id INTEGER PRIMARY KEY, doc INTEGER, '
                'term INTEGER, score FLOAT)')
    cur.execute('CREATE INDEX doc_term_idx1 ON doc_term(doc)')
    cur.execute('CREATE INDEX doc_term_idx2 ON doc_term(term)')
    con.commit()

    for doc_no, doc in enumerate(file(wordcount_file, 'r')):
        doc = doc.split()[1:]
        terms = {}
        for term in doc:
            terms[int(term.split(':')[0])] = int(term.split(':')[1])

        keys = terms.keys()


        res = generic_generator((doc_no,)*len(keys),
                                keys, [terms[i] for i in keys])
        execution_str = 'INSERT INTO doc_term (id, doc, term, score) '
        execution_str += 'VALUES(NULL, ?, ?, ?)'
        cur.executemany(execution_str, res)

    con.commit()

def write_terms(con, cur, terms_file):
    cur.execute('CREATE TABLE terms (id INTEGER PRIMARY KEY, title VARCHAR(100))')
    con.commit()

    res = file_generator(open(terms_files, 'r'))
    cur.executemany('INSERT INTO terms (id, title) VALUES(NULL, ?)',
                    map(buffer, res))
    con.commit()

def write_docs(con, cur, docs_file):
    cur.execute('CREATE TABLE docs (id INTEGER PRIMARY KEY, title VARCHAR(100))')
    con.commit()

    res = docs_file #HACK: my docs_file is already a generator
    cur.executemany('INSERT INTO docs (id, title) VALUES(NULL, ?)',
                    map(buffer, res))

    con.commit()


### My own iterable corpora instead of a documents file ###

from pymongo import Connection
from pymongo import ASCENDING, DESCENDING

connection = Connection()
db = connection.db
stemmed_documents = db.stemmed_documents

class MyCorpus(object):
    """
    A gensim corpus object

    Notes
    -----
    This defines an __iter__ method. The corpus is always iterated through
    by filename sorted in ascending order.
    """
    def __init__(self):
        pass

    def __iter__(self):
        # if you need to, create an index for the database by doing
        # documents.create_index([("filename", ASCENDING)])
        # stemmed_documents.create_index([("filename", ASCENDING)])
        for doc in stemmed_documents.find(sort=[('filename', ASCENDING)]):
            yield doc["stemmed_text"]

### main ###

if (__name__ == '__main__'):
    if (len(sys.argv) != 6): # hack, using on doc-file object
       print 'usage: python generate_csvs.py <db-filename> <doc-wordcount-file> <beta-file> <gamma-file> <vocab-file> <doc-file>\n'
       sys.exit(1)

    filename = sys.argv[1]
    doc_wordcount_file = sys.argv[2]
    beta_file = sys.argv[3]
    gamma_file = sys.argv[4]
    vocab_file = sys.argv[5]
    #doc_file = sys.argv[6]
    doc_file = MyCorpus()

    # connect to database, which is presumed to not already exist
    con = sqlite3.connect(filename)
    cur = con.cursor()

    # pre-process vocab, since several of the below functions need it in this format
    vocab = file(vocab_file, 'r').readlines()
    vocab = map(lambda x: x.strip(), vocab)

    # write the relevant rlations to the database, see individual functions for details
    print "writing terms to db..."
    write_terms(con, cur, vocab_file)

    print "writing docs to db..."
    write_docs(con, cur, doc_file)

    print "writing doc_doc to db..."
    write_doc_doc(con, cur, gamma_file)

    print "writing doc_topic to db..."
    write_doc_topic(con, cur, gamma_file)

    print "writing topics to db..."
    write_topics(con, cur, beta_file, vocab)

    print "writing topic_term to db..."
    write_topic_term(con, cur, beta_file)

    print "writing topic_topic to db..."
    write_topic_topic(con, cur, beta_file)

    print "writing term_term to db..."
    write_term_term(con, cur, beta_file, len(vocab))

    print "writing doc_term to db..."
    write_doc_term(con, cur, doc_wordcount_file, len(vocab))

