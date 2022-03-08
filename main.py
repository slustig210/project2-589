from utils import question, QUESTIONS
import matplotlib.pyplot as plt
from naive_bayes import naive_bayes


def boxedPrint(s: str, boxChar: str = '*'):
    assert len(boxChar) == 1

    print(boxChar * (len(s) + 4))
    print(f"{boxChar} {s} {boxChar}")
    print(boxChar * (len(s) + 4))


@question
def question1():
    boxedPrint("Question 1")
    naive_bayes(useLog=False, alpha=0)
    naive_bayes(alpha=0)


@question
def question2():
    boxedPrint("Question 2")
    naive_bayes()

    x, y = [], []
    alpha = 0.0001
    while alpha <= 1000:
        x.append(alpha)
        print(f"Running {alpha = }")
        res = naive_bayes(alpha=alpha, output=False)
        y.append((res[0] + res[3]) / sum(res))
        alpha *= 10

    plt.xscale("log")
    plt.plot(x, y)

    plt.show()


@question
def question3():
    boxedPrint("Question 3")
    naive_bayes(1, 1, 1, 1, alpha=10)


@question
def question4():
    boxedPrint("Question 4")
    naive_bayes(0.5, 0.5, 1, 1, alpha=10)


@question(6)
def question6():
    boxedPrint("Question 6")
    naive_bayes(0.1, 0.5, 1, 1)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        QUESTIONS[arg]()