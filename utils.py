import re
import os
import glob
import random
from typing import Callable, overload
from nltk.corpus import stopwords
import nltk

REPLACE_NO_SPACE = re.compile(r"[._;:!`¦\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile(r"(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords')


def preprocess_text(text: str):
    stop_words = set(stopwords.words('english'))
    text = REPLACE_NO_SPACE.sub("", text)
    text = REPLACE_WITH_SPACE.sub(" ", text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = text.split()
    return [w for w in words if w not in stop_words]


def load_training_set(percentage_positives, percentage_negatives):
    vocab: set[str] = set()
    positive_instances: list[list[str]] = []
    negative_instances: list[list[str]] = []
    for filename in glob.glob('train/pos/*.txt'):
        if random.random() > percentage_positives:
            continue
        with open(os.path.join(os.getcwd(), filename), 'r',
                  encoding="utf-8") as f:
            contents = f.read()
            contents = preprocess_text(contents)
            positive_instances.append(contents)
            vocab = vocab.union(set(contents))
    for filename in glob.glob('train/neg/*.txt'):
        if random.random() > percentage_negatives:
            continue
        with open(os.path.join(os.getcwd(), filename), 'r',
                  encoding="utf-8") as f:
            contents = f.read()
            contents = preprocess_text(contents)
            negative_instances.append(contents)
            vocab = vocab.union(set(contents))
    return positive_instances, negative_instances, vocab


def load_test_set(percentage_positives, percentage_negatives):
    positive_instances: list[list[str]] = []
    negative_instances: list[list[str]] = []
    for filename in glob.glob('test/pos/*.txt'):
        if random.random() > percentage_positives:
            continue
        with open(os.path.join(os.getcwd(), filename), 'r',
                  encoding="utf-8") as f:
            contents = f.read()
            contents = preprocess_text(contents)
            positive_instances.append(contents)
    for filename in glob.glob('test/neg/*.txt'):
        if random.random() > percentage_negatives:
            continue
        with open(os.path.join(os.getcwd(), filename), 'r',
                  encoding="utf-8") as f:
            contents = f.read()
            contents = preprocess_text(contents)
            negative_instances.append(contents)
    return positive_instances, negative_instances


QUESTIONS = dict[str, Callable[[], None]]()


@overload
def question(num: int) -> Callable[[Callable[[], None]], Callable[[], None]]:
    ...


@overload
def question(f: Callable[[], None]) -> Callable[[], None]:
    ...


def question(arg: int | Callable[[], None]):
    """
    `question: (num: int) -> ((() -> None) -> (() -> None))`
    
    Returns a decorator to add the question with the given number
    to the QUESTIONS dict.

    Args:
        num (int): The question number

    Returns:
        (() -> None) -> (() -> None): The decorator.

    Usage:

    >>> @question(4)
    ... def myQuestion():
    ...     ...
    ... 
    >>> QUESTIONS['4'] == myQuestion
    True

    --------------------------------------------------------------------

    `question: (f: () -> None) -> (() -> None)`

    Decorator to add the next question to the QUESTIONS dict.
    The key will be one plus the last question; the last question is the maximum
    question number so far.

    Args:
        f (() -> None): The function corresponding to the next question.

    Returns:
        f, unmodified
    """
    mx: int = getattr(question, "mx", 0)

    if isinstance(arg, int):

        def decorator(f: Callable[[], None]):
            QUESTIONS[str(arg)] = f
            question.mx = max(arg, mx)
            return f

        return decorator

    question.mx = mx + 1
    QUESTIONS[str(question.mx)] = arg
    return arg


def boxedPrint(s: str, boxChar: str = '*'):
    """Print the inputted string with a box around it.

    Args:
        s (str): The string to print.
        boxChar (str, optional): The character to create the box with. Must be length 1. Defaults to '*'.

    Usage:

        >>> boxedPrint("Hello, World!", '=') 
        =================
        = Hello, World! =
        =================
        >>> boxedPrint("Question 1")
        **************
        * Question 1 *
        **************
    """

    assert len(
        boxChar) == 1, "boxChar must be a single character with length 1."

    print(boxChar * (len(s) + 4))
    print(f"{boxChar} {s} {boxChar}")
    print(boxChar * (len(s) + 4))