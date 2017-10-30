from __future__ import print_function, absolute_import

import os
import argparse
import numpy as np
import scipy.stats


def evaluate_vectors_analogy(W, vocab, ivocab):
    """Evaluate the trained word vectors on a variety of tasks"""

    print("Analogy Task")

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
    ]
    prefix = os.path.join(os.path.dirname(__file__), "data", "eval", "question-data")

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0  # count correct semantic questions
    correct_syn = 0  # count correct syntactic questions
    correct_tot = 0  # count correct questions
    count_sem = 0    # count all semantic questions
    count_syn = 0    # count all syntactic questions
    count_tot = 0    # count all questions
    full_count = 0   # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j * split_size, min((j + 1) * split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :] + W[ind3[subset], :])
            # cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions)  # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("    %s:" % filenames[i][:-4])
        print('        ACCURACY TOP1: %.2f%% (%d/%d)' %
              (np.mean(val) * 100, np.sum(val), len(val)))

    print('    Questions seen/total: %.2f%% (%d/%d)' %
          (100 * count_tot / float(full_count), count_tot, full_count))
    print('    Semantic accuracy: %.2f%%  (%i/%i)' %
          (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print('    Syntactic accuracy: %.2f%%  (%i/%i)' %
          (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)\n' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


def evaluate_vectors_sim(W, vocab, ivocab):
    """Evaluate the trained word vectors on the WordSimilarity-353 task."""

    filename = 'combined.csv'
    # filename = 'set1.csv'
    filename = os.path.join(os.path.dirname(__file__), "data", "eval", "wordsim353", filename)

    with open(filename, 'r') as f:
        data = [line.rstrip().split(',') for line in f][1:]

    # TODO: include cases where words are missing
    data = [row for row in data if (row[0] in vocab and row[1] in vocab)]
    words = np.array([[vocab[row[0]], vocab[row[1]]] for row in data])
    score = np.array([float(row[2]) for row in data])
    pred = np.sum(np.multiply(W[words[:, 0], :], W[words[:, 1], :]), 1)

    rho, p = scipy.stats.spearmanr(score, pred)
    print("WordSimilarity-353 Spearman Correlation: %.3f\n" % rho)


def evaluate_human_sim():
    """Evaluate the trained word vectors on the WordSimilarity-353 task."""

    filename = 'set1.csv'
    filename = os.path.join(os.path.dirname(__file__), "data", "eval", "wordsim353", filename)

    with open(filename, 'r') as f:
        data = [line.rstrip().split(',') for line in f][1:]

    # TODO: include cases where words are missing
    mean = np.array([float(row[2]) for row in data])
    score = np.array([[float(row[i]) for i in range(3, len(row))] for row in data])

    n, m = score.shape
    trials = 100
    total = 0.
    for i in range(trials):
        group = np.zeros(m, np.bool)
        group[np.random.choice(m, m / 2, False)] = True
        score1 = np.mean(score[:, group], 1)
        score2 = np.mean(score[:, np.invert(group)], 1)

        rho, p = scipy.stats.spearmanr(score1, score2)
        total += rho
    print("Human WordSimilarity-353 Spearman Correlation: %.3f\n" % (total / trials))
