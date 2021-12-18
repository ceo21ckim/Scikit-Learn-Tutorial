
# Ridge coefficients as a function of the regularization

from sklearn.linear_model import Ridge 

# alpha = .5
reg = Ridge(alpha = .5)

reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

reg.coef_ 
# array([0.34545455, 0.34545455])

reg.intercept_
# 0.1363636....

import numpy as np 
import matplotlib.pyplot as plt

x = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

alphas

coefs = []

for a in alphas:
    ridge = Ridge(alpha = a, fit_intercept=False)
    ridge.fit(x, y)
    coefs.append(ridge.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


# Classification of text documents using sparse features

import logging 
import numpy as np 
from optparse import OptionParser 
import sys 
from time import time 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer 
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2 
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline 
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density 
from sklearn import metrics 

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s')

op = OptionParser()
op.add_option(
    "--report", 
    action = 'store_true', 
    dest = 'print_report', 
    help = 'Print a detailes classification report.',
)

op.add_option(
    "--chi2_select", 
    action = 'store', 
    type = 'int',
    dest = 'select_chi2', 
    help = 'Select some number of features using a chi-sqaured test.',
)

op.add_option(
    '--confustion_matrix', 
    action = 'store_true', 
    dest = 'print_cm', 
    help = 'Print the confusion matrix.',
)

op.add_option(
    '--top10', 
    actions = 'store_true', 
    dest = 'print_top10', 
    help = 'Print ten most discriminative terms per class for every classifier.',    
)

op.add_option(
    '--all_categories', 
    action = 'store_true', 
    dest = 'all_categories', 
    help = 'Whether to use all categories or not.',
)

op.add_option(
    '--use_hashing', 
    action = 'store_true', 
    help = 'Use a hashing vectorizer.'
)

op.add_option(
    '--n_features', 
    action = 'store', 
    type = int, 
    default = 2 ** 16, 
    help = 'n_features when using the hashing vectorizer.'
)

op.add_option(
    "--filtered", 
    action = 'store_true', 
    help = (
        "remove newsgroup information that is easily overfit: "
        "hedaers, signatures, and quoting."
    ),
)

def is_interactive():
    return not hasattr(sys.modules["__main__"], "__file__")

argv = [] if is_interactive() else sys.argv[1:]

(opts, args) = op.parse_args(argv)

if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)
    
print(__doc__)
op.print_help()
print()



# Load data from the training set 

if opts.all_categories:
    categories = None 
else: 
    categories = [
        'alt.atheism', 
        'talk.religion.misc', 
        'comp.graphics', 
        'sci.space'
    ]
    
if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()
    

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else 'all')

data_train = fetch_20newsgroups(
    subset = 'train', categories = categories, shuffle = True, random_state = 42, remove = remove
)

data_test = fetch_20newsgroups(
    subset = 'test', categories = categories, shuffle = True, random_state = 42, remove = remove
)

target_names = data_train.target_names

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print(
    '%d documents - %0.3fMB (training set)' % (len(data_train.data), data_train_size_mb)
)
print("%d documents - %0.3fMB (test set)" % (len(data_test.data), data_test_size_mb))
print("%d categories" % len(target_names))
print()

y_train, y_test = data_train.target, data_test.target 

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if opts.use_hashing :
    vectorizer = HashingVectorizer(
        stop_words = 'english', alternate_sign = False, n_features = opts.n_features
    )
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = 'english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples : %d, n_features : %d" % X_train.shape)
print()

print("Extracting features from the test data using a same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples : %d, n_features : %d" % X_test.shape)
print()

# mapping from interger feature name to original token string
if opts.use_hashing:
    feature_names = None 
else:
    feature_names = vectorizer.get_feature_names_out()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" % opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k = opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names is not None:
        feature_names = feature_names[ch2.get_support()]
        
    print("done in %fs " % (time() - t0))
    print()
    
def trim(s):
    return s if len(s) <= 80 else s[:77] + '...'

def benchmark(clf):
    print("_" * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, "coef_"):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred, target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split("(")[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
    (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
    (Perceptron(max_iter=50), "Perceptron"),
    (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
    (KNeighborsClassifier(n_neighbors=10), "kNN"),
    (RandomForestClassifier(), "Random forest"),
):
    print("=" * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print("=" * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=0.0001, max_iter=50, penalty=penalty)))

# Train SGD with Elastic Net penalty
print("=" * 80)
print("Elastic-Net penalty")
results.append(
    benchmark(SGDClassifier(alpha=0.0001, max_iter=50, penalty="elasticnet"))
)

# Train NearestCentroid without threshold
print("=" * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print("=" * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=0.01)))
results.append(benchmark(BernoulliNB(alpha=0.01)))
results.append(benchmark(ComplementNB(alpha=0.1)))

print("=" * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(
    benchmark(
        Pipeline(
            [
                (
                    "feature_selection",
                    SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3)),
                ),
                ("classification", LinearSVC(penalty="l2")),
            ]
        )
    )
)



indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, 0.2, label="score", color="navy")
plt.barh(indices + 0.3, training_time, 0.2, label="training time", color="c")
plt.barh(indices + 0.6, test_time, 0.2, label="test time", color="darkorange")
plt.yticks(())
plt.legend(loc="best")
plt.subplots_adjust(left=0.25)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(bottom=0.05)

for i, c in zip(indices, clf_names):
    plt.text(-0.3, i, c)

plt.show()