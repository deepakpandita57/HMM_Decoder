# Implement the Hidden Markov Model (HMM) decoder using train and test data and report the accuracy on the test file.


Instructions for running "HMMDecoder.py"
=============================================================================================================
To run the script "HMMDecoder.py" change the values of "train_file" and "test_file" variables in the script.

Description
=============================================================================================================
I have calculated the transition and emission probabilities (log probability) from the train file.
Then, I have used these probabilities to find the best tag sequence for a given sentence in the test file.
Viterbi algorithm is implemented in the function "viterbi".


Accuracy
=============================================================================================================
Correct tags: 52008
Total tags: 56684
Accuracy: 91.75%


References
=============================================================================================================
This was done as a homework problem in the Statistical Speech and Language Processing class (CSC 448, Fall 2017) by Prof. Daniel Gildea (https://www.cs.rochester.edu/~gildea/) at the University of Rochester, New York.