# Importing the training and testing data

Import the training and testing data into folders `train` and `test` in
the root folder of this project. These should both contain a folder `pos`
and a folder `neg`, each filled with `.txt` files containing either positive
or negative movie reviews respectively.

# Installing dependencies

The required dependencies are `matplotlib` and `nltk`. Install these using `pip`.
Also the python version should be `python3` (I am using `3.10.2`, I believe `3.9` should
work as well (?))

# Running the code

Run `python3 main.py` to run the program for all questions.
Run `python3 main.py [question number(s)]` to run the code for specific questions.
For example, to run question 1 and then question 3, you can run
`python3 main.py 1 3`.

Note that question 2 specifically plots using `matplotlib`, so you will need to
close that to continue execution.