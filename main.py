from utils import question, QUESTIONS, boxedPrint
import matplotlib.pyplot as plt
from naive_bayes import naive_bayes


@question(1)
def question1():
    boxedPrint("Question 1")
    naive_bayes(useLog=False, alpha=0)
    naive_bayes(alpha=0)


@question(2)
def question2():
    boxedPrint("Question 2")
    naive_bayes(alpha=1)

    x, y = list[float](), list[float]()
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


@question(3)
def question3():
    boxedPrint("Question 3")
    naive_bayes(1, 1, 1, 1)


@question(4)
def question4():
    boxedPrint("Question 4")
    naive_bayes(0.5, 0.5, 1, 1)


@question(6)
def question6():
    boxedPrint("Question 6")
    naive_bayes(0.1, 0.5, 1, 1)


if __name__ == "__main__":
    print()

    import sys

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            try:
                QUESTIONS[int(arg)]()
            except ValueError:
                print("Please provide integer arguments.")
            except KeyError:
                print(f"No such question number {arg}.")
    else:
        for q in QUESTIONS.values():
            q()