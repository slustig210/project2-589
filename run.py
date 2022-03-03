from utils import preprocess_text, load_training_set, load_test_set
import pprint
from collections import Counter
from math import prod, log2


def naive_bayes(logProb: bool = True):
    percentage_positive_instances_train = 0.0004
    percentage_negative_instances_train = 0.0004

    percentage_positive_instances_test = 0.0004
    percentage_negative_instances_test = 0.0004

    (pos_train, neg_train,
     vocab) = load_training_set(percentage_positive_instances_train,
                                percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test,
                                         percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    with open('vocab.txt', 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)
    print("Vocabulary (training set):", len(vocab))

    # probability positive = len(pos_train) / (len(pos_train) + len(neg_train))
    assert len(pos_train) + len(neg_train) > 0, \
            f"{len(pos_train) = }, {len(neg_train) = }"

    prPos = len(pos_train) / (len(pos_train) + len(neg_train))
    prNeg = 1 - prPos

    # Counts of each word in the vocabulary
    # assuming all words in the documents are in the vocab
    posCounts = Counter(w for doc in pos_train for w in doc)
    negCounts = Counter(w for doc in neg_train for w in doc)

    totalPosCounts = sum(posCounts.values())
    totalNegCounts = sum(negCounts.values())

    # Pr(w | pos) = posCounts[w] / totalPosCounts
    # Pr(w | neg) = negCounts[w] / totalNegCounts

    if logProb:
        truePos = 0
        for doc in pos_test:
            pos = prPos * prod(posCounts[w] / totalPosCounts for w in doc)
            neg = prNeg * prod(negCounts[w] / totalNegCounts for w in doc)

            if pos >= neg:
                truePos += 1

        trueNeg = 0
        for doc in neg_test:
            pos = prPos * prod(posCounts[w] / totalPosCounts for w in doc)
            neg = prNeg * prod(negCounts[w] / totalNegCounts for w in doc)

            if pos < neg:
                trueNeg += 1
    else:
        # TODO: not sure if supposed to add 1 to all the counts... check the slides ig
        truePos = 0
        for doc in pos_test:
            pos = log2(prPos) + sum(
                log2(posCounts[w] / totalPosCounts)
                for w in doc
                if posCounts[w])
            neg = log2(prNeg) + sum(
                log2(negCounts[w] / totalNegCounts)
                for w in doc
                if negCounts[w])

            if pos >= neg:
                truePos += 1

        trueNeg = 0
        for doc in neg_test:
            pos = log2(prPos) + sum(
                log2(posCounts[w] / totalPosCounts)
                for w in doc
                if posCounts[w])
            neg = log2(prNeg) + sum(
                log2(negCounts[w] / totalNegCounts)
                for w in doc
                if negCounts[w])

            if neg < pos:
                trueNeg += 1


if __name__ == "__main__":
    naive_bayes()
