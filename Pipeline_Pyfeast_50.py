'''Feature selection (PyFeast)

Mutual Information Maximisation
Joint Mutual Information
Mutual Information Feature Selection
Conditional Infomax Feature Extraction			cant find
Max-Relevance Min-Redundancy -----------------
Conditional Mutual Info Maximisation			can't find
Conditional Mutual Information 					can't find
Conditional Redundancy
Double Input Symmetrical Relevance
Interaction Capping
'''

#print(__doc__)


# Code source: Gael Varoquaux
#              Andreas Muller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition, ensemble, random_projection, cluster, feature_selection, pipeline, svm, linear_model
from sklearn import cross_validation
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, LogisticRegression, Ridge, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD, DictionaryLearning, FactorAnalysis, SparsePCA, NMF, PCA, RandomizedPCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
import csv
from feast import *
h = .02  # step size in the mesh

pct_features_list = [0.5]

pyfeast_names = [
"MIM",
"JMI",
"MIFS",
"CIFE",
"mRMR",
"CMIM",
"CondMI",
"Condred",
"DISR",
"ICAP"
]

dim_red_names = [
"LDA",
"Truncated SVD",
#"Sparse Coder",
"Dictionary Learning",
"Factor Analysis",
"Sparse PCA",
"Non-Negative Matrix Factorization (reg)",
"Exact PCA",
"Randomized PCA",
"Kernel PCA - Linear",
"Kernel PCA - Polynomial",
"Kernel PCA - RBF",
"Kernel PCA - Sigmoid",
"Kernel PCA - Cosine",
"Isomap",
"LLE",
"Modified LLE",
"Local Tangent Space Alignment",
"Spectral Embedding"
]



def get_dim_reds_pyfeast(pct_features):
	n_components=max(int(pct_features * num_features), 1)
	return [
	MIM(X,Y, n_components),
	JMI(X,Y, n_components),
	MIFS(X,Y, n_components),
	CIFE(X,Y, n_components),
	mRMR(X,Y, n_components),
	CMIM(X,Y, n_components),
	CondMI(X,Y, n_components),
	Condred(X,Y, n_components),
	DISR(X,Y, n_components),
	ICAP(X,Y, n_components)
	]

'''MUST initialize data first'''
def get_dim_reds_scikit(pct_features):
	n_components = max(int(pct_features * num_features), 1)
	return [
	LinearDiscriminantAnalysis(n_components=n_components),
	TruncatedSVD(n_components=n_components),
	#SparseCoder(n_components=n_components),
	DictionaryLearning(n_components=n_components),
	FactorAnalysis(n_components=n_components),
	SparsePCA(n_components=n_components),
	NMF(n_components=n_components),
	PCA(n_components=n_components),
	RandomizedPCA(n_components=n_components),
	KernelPCA(kernel="linear", n_components=n_components),
	KernelPCA(kernel="poly", n_components=n_components),
	KernelPCA(kernel="rbf", n_components=n_components),
	KernelPCA(kernel="sigmoid", n_components=n_components),
	KernelPCA(kernel="cosine", n_components=n_components),
	Isomap(n_components=n_components),
	LocallyLinearEmbedding(n_components=n_components, eigen_solver='auto', method='standard'),
	LocallyLinearEmbedding(n_neighbors=n_components, n_components=n_components, eigen_solver='auto', method='modified'),
	LocallyLinearEmbedding(n_neighbors=n_components, n_components=n_components, eigen_solver='auto', method='ltsa'),
	SpectralEmbedding(n_components=n_components)
	]

def select_data_indeces(indeces):
	return X[indeces,:]

classifier_names = ["Perceptron",
		 "Logistic Regression",
		 "Linear Discriminant Analysis",
		  "Ridge",
		 "Passive Aggressive", 
		 "Linear SVM", 
		 #"Polynomial SVM", 
		 #"RBF SVM", 
		 #"Sigmoid SVM"
		 "Nearest Neighbors", 
		 "Quadratic Discriminant Analysis", 
		 "Bernoulli Naive Bayes", "Decision Tree",
         "Random Forest", 
         "AdaBoost", 
         "Bagging", 
         "Gradient Boosting", 
         "Voting"]

classifiers = [
	Perceptron(),
	LogisticRegression(),
    LinearDiscriminantAnalysis(),
    Ridge(),
    PassiveAggressiveClassifier(),
    LinearSVC(C=0.025),
#    SVC(kernel="poly", C=0.025),
#    SVC(gamma=2, C=1), #RBF SVM
#    SVC(kernel="sigmoid", C=0.025),
    KNeighborsClassifier(3),
    QuadraticDiscriminantAnalysis(),
    BernoulliNB(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    BaggingClassifier(),
    GradientBoostingClassifier(),
    VotingClassifier(estimators=[
    	('Linear SVM', LinearSVC(C=0.025)),
    	('Decision Tree', DecisionTreeClassifier(max_depth=5)),
    	('KNeighbors', KNeighborsClassifier(3)),
    	('Bernoulli Naive Bayes', BernoulliNB()),
    	('Logistic Regression', LogisticRegression())
    	])
    ]

X = None
Y = None



with open("data_full.csv") as infile:
	global X, Y
	reader = csv.reader(infile, delimiter='\t', quotechar="|")
	X = []
	Y = []
	for i,row in enumerate(reader):
		X.append([float(n) for n in row[:-2]])
		Y.append(int(row[-1]))
	X = np.matrix(X)
	Y = np.array(Y)



'''
with open("segmentation_fix_pos.vcf") as posfile, open("segmentation_fix_neg.vcf") as negfile:
	global X,Y
	posreader = csv.reader(posfile, delimiter='\t', quotechar="|") 
	negreader = csv.reader(negfile, delimiter='\t', quotechar="|")
	X = []
	Y = []
	posreader.next()
	negreader.next()
	for i in range(50):
		X.append([float(n) for n in posreader.next()[4:]])
		Y.append(1)
	for i in range(50):
		X.append([float(n) for n in negreader.next()[4:]])
		Y.append(0)

	posfile.close()
	negfile.close()

	outfile = open('data.csv', 'wb')
	for i,row in enumerate(X):
		outfile.write('\t'.join(map(str,row)) + '\t' + str(Y[i]) + '\n')
	X = np.matrix(X)
	Y = np.array(Y)
	print X.shape
'''

out = open("output_pyfeast_50.csv", "wb")
out.write("% Features, Reduction Technique, Classifier, Score, Duration, Error\n")
out.flush()
(num_data_points, num_features) = X.shape

import time

for pct_features in pct_features_list:
	dim_reds = get_dim_reds_pyfeast(pct_features)
	for (dim_red_name, dim_red_list) in zip(pyfeast_names, dim_reds):
		dim_red_data = X[:,map(int,dim_red_list)]
		for (classifier_name, classifier) in zip(classifier_names, classifiers):
			start = time.time()
			try:
				print dim_red_name + " " + classifier_name + " starting"
				scores = cross_validation.cross_val_score(classifier, dim_red_data, Y, cv=10)
				end = time.time()
				duration = (end - start) / 60.0
				#out.write("Reduction Technique: %s, Classifier: %s, Score: %.4f \n" % (dim_red_name, classifier_name, scores.mean()))
				out.write(",".join([str(pct_features), dim_red_name, classifier_name, str(scores.mean()), str(duration), ""]) + "\n")
				print(",".join([str(pct_features), dim_red_name, classifier_name, str(scores.mean()), str(duration) + " min", ""]) + "\n")
				out.flush()    
			except ValueError as e:
				end = time.time()
				duration = (end - start) / 60.0
                #out.write("ERROR: Reduction Technique: %s, Classifier: %s, Error: %s \n" % (dim_red_name, classifier_name, e))
				out.write(",".join([str(pct_features), dim_red_name, classifier_name, "", str(duration), e]) + "\n")
				print(",".join([str(pct_features), dim_red_name, classifier_name, "", str(duration) + " min", e]) + "\n")
				out.flush()

