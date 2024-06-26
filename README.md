# Sentiment-Analysis

Classifying Movie reviews from the famous [SST dataset](https://huggingface.co/datasets/stanfordnlp/sst?library=true).
In this project We assume 5 classes according to the float score label that indicates the level of positive sentiment from 0.0 to 1.0.
The mapping from scores to classes is summarized in the below table. assume all upper bounds are inclusive , all lower bounds are non inclusive except for class 0
| Range     | Class | Description     |
|-----------|-------|-----------------|
| 0.0-0.2   | 0     | Very Negative   |
| 0.2-0.4   | 1     | Negative        |
| 0.4-0.6   | 2     | Neutral         |
| 0.6-0.8   | 3     | Positive        |
| 0.8-1.0   | 4     | Very Positive   |
 

I use Naive Bayes and Logistic regression classifiers built from scratch using only numpy.
Then I compare my results to scikit-learn's implementation of [MultiNomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html), [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html), and [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

The Naive Bayes classifier is based on the pseudocode from  Dan Jurafsky and James H. Martin's [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). 
chapter 4.

 # Achieved exactly Identical results to scickit-learn's MultiNomialNB.

The Logistic regression implementation was also loosely based on chapter 5 of the same textbook.

I used simple bigram features for training and testing As an example, the sentence “I love this movie very much” has 5 word bi-gram features namely (‘I’, ‘love’), (‘love’, ‘this’) and so on. Each sentence is represented with a vector of length equal to the number of unique word bi-grams in the whole dataset with 1 at the corresponding index if the bi-gram exists and 0 otherwise.
I used Stochastic Gradient descent for optimization and no regularization .


in the `logistic_regression_from_scratch_and_SGDcalsifier` notebook I compare the results of my implementaion with SGDcLassifier.
The results are nearly identical with only slight diffrences likely due to numerical precision reasons.

The last notebook `logistic_regression_sklearn` shows the results of running the same dataset on scickit-learn's LogisticRegression. 
Note that these results are different from mine since this class uses a different optimizer : [lbfgs](https://en.wikipedia.org/wiki/Limited-memory_BFGS).
refer to [scickit-learn's documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for more information.



