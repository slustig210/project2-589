from utils import load_training_set, load_test_set
# import pprint
from collections import Counter
from math import inf, prod, log2


def naive_bayes(percentage_positive_instances_train: float = 0.0004,
                percentage_negative_instances_train: float = 0.0004,
                percentage_positive_instances_test: float = 0.0004,
                percentage_negative_instances_test: float = 0.0004,
                useLog: bool = True,
                alpha: int = 1):
    """Train the naive bayes algorithm on a randomly chosen training set
    and then test on a randomly chosen testing set.

    Args:
        percentage_positive_instances_train (float, optional): The amount of positive instances to train with. Defaults to 0.0004.
        percentage_negative_instances_train (float, optional): The amount of negative instances to train with. Defaults to 0.0004.
        percentage_positive_instances_test (float, optional): The amount of positive instances to test with. Defaults to 0.0004.
        percentage_negative_instances_test (float, optional): The amount of negative instances to test with. Defaults to 0.0004.
        useLog (bool, optional): Whether or not to compute using log probabilities to improve accuracy. Defaults to True.
        alpha (int, optional): Value of alpha for Laplace smoothing. Defaults to 1.

    Returns:
        tuple[int, int, int, int]: truePos, falseNeg, falsePos, trueNeg
    """

    assert 0 <= percentage_positive_instances_train <= 1 and \
        0 <= percentage_negative_instances_train <= 1 and \
        0 <= percentage_positive_instances_test <= 1 and \
        0 <= percentage_negative_instances_test <= 1

    assert alpha >= 0

    print("Loading instances...")

    pos_train, neg_train, vocab = load_training_set(
        percentage_positive_instances_train,
        percentage_negative_instances_train)

    pos_test, neg_test = load_test_set(percentage_positive_instances_test,
                                       percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    assert len(pos_train) + len(neg_train) > 0, \
            f"{len(pos_train) = }, {len(neg_train) = }"

    print(f"Running with{'' if useLog else 'out'} log probabilities")
    print(f"{alpha = }")

    alphaV = alpha * len(vocab)

    # probability positive = len(pos_train) / (len(pos_train) + len(neg_train))
    prPos = len(pos_train) / (len(pos_train) + len(neg_train))
    prNeg = 1 - prPos

    # Counts of each word in the vocabulary
    posCounts = Counter(w for doc in pos_train for w in doc)
    negCounts = Counter(w for doc in neg_train for w in doc)

    totalPosCounts = sum(posCounts.values())
    totalNegCounts = sum(negCounts.values())

    # Pr(w | pos) = posCounts[w] / totalPosCounts
    # Pr(w | neg) = negCounts[w] / totalNegCounts

    if useLog:

        def evaluateDoc(doc: list[str]):
            try:
                pos = log2(prPos) + sum(
                    log2((posCounts[w] + alpha) / (totalPosCounts + alphaV))
                    for w in set(doc))
            except ValueError:
                pos = -inf

            try:
                neg = log2(prNeg) + sum(
                    log2((negCounts[w] + alpha) / (totalNegCounts + alphaV))
                    for w in set(doc))
            except ValueError:
                neg = -inf

            return pos >= neg
    else:

        def evaluateDoc(doc: list[str]):
            pos = prPos * prod(
                (posCounts[w] + alpha) / (totalPosCounts + alphaV)
                for w in set(doc))
            neg = prNeg * prod(
                (negCounts[w] + alpha) / (totalNegCounts + alphaV)
                for w in set(doc))

            return pos >= neg

    truePos = sum(1 for doc in pos_test if evaluateDoc(doc))
    falsePos = sum(1 for doc in neg_test if evaluateDoc(doc))

    trueNeg, falseNeg = len(neg_test) - falsePos, len(pos_test) - truePos

    print("Confusion matrix:")
    print(f"{truePos:<12}{falseNeg}\n{falsePos:<12}{falseNeg}")

    print("Accuracy:", (truePos + falsePos) / (len(pos_test) + len(neg_test)))
    print("Precision:", truePos / (truePos + falsePos))
    print("Recall:", truePos / (truePos + falseNeg))

    return truePos, falseNeg, falsePos, trueNeg


def boxedPrint(s: str, boxChar: str = '*'):
    assert len(boxChar) == 1

    print(boxChar * (len(s) + 4))
    print(f"{boxChar} {s} {boxChar}")
    print(boxChar * (len(s) + 4))


def question1():
    # Question 1
    boxedPrint("Question 1")
    naive_bayes(0.2, 0.2, 0.2, 0.2, False, 0)


def main():
    question1()


if __name__ == "__main__":
    main()
