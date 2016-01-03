from time import time
import numpy
from sklearn import manifold, datasets, decomposition, random_projection, cluster, feature_selection, pipeline, svm, linear_model

pct_features_list = [0.2, 0.4, 0.6, 0.8, 0.9];


def import_data():
	import csv
	with open('all_combined.csv', 'rb') as f:
		reader = csv.reader(f)
		data = list(reader)

		#Num sequences, num features
		print len(data), len(data[0])
		for x in range(len(data)):
			data[x] = data[x][4:]

		#remove title rows
		data = data[1:]
		array = numpy.array(data).astype(float)
		f.close()
		return array

def LDA(array, test_labels):
	
	#LDA
	from sklearn.lda import LDA

	print "LDA"
	print "Features\tTime" 

	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		LDA(n_components=num_features).fit(array,test_labels)
		end = time()
		print num_features, "\t", (end - start)

def Isomap(array, percent_samples):

	print "Isomap"
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = manifold.Isomap(n_components=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def LLE(array, percent_samples):
	
	print "LLE with", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = manifold.LocallyLinearEmbedding(n_components=num_features, eigen_solver='auto', method='standard').fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def ModifiedLLE(array, percent_samples):
	print "Modified LLE with", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = manifold.LocallyLinearEmbedding(n_neighbors=num_features, n_components=num_features, eigen_solver='auto', method='modified').fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def Hessian(array, percent_samples):
	print "Hessian LLE with", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		#n_neighbors must be >= n_components, so here they are set equal
		Y = manifold.LocallyLinearEmbedding(n_neighbors=num_features * (num_features + 3) / 2 + 1, n_components=num_features, eigen_solver='auto', method='hessian').fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def SpectralEmbedding(array, percent_samples):
	print "Spectral Embedding with", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = manifold.SpectralEmbedding(n_components=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def LTSA(array, percent_samples):
	print "Local Tangent Space Alignment with", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = manifold.LocallyLinearEmbedding(n_neighbors=num_features, n_components=num_features, eigen_solver='auto', method='ltsa').fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def MDSMetric(array, percent_samples):
	print "Metric Multi-dimensional Scaling with", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = manifold.MDS(num_features, max_iter=100, n_init=1).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def MDSNonmetric(array, percent_samples):
	print "Nonmetric Multi-dimensional Scaling with", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = manifold.MDS(num_features, max_iter=100, n_init=1, metric=False).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def TSNE(array, percent_samples):
	print "t-distributed Stochastic Neighbor Embedding", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = manifold.TSNE(n_components=num_features, init='pca', random_state=0).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def PCA(array, percent_samples):
	print "Exact PCA", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = decomposition.PCA(n_components=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def KernelPCA(array, percent_samples):
	print "Kernel PCA", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	kernels = ["linear", "poly", "rbf", "sigmoid", "cosine"]
	for method in kernels:
		print "\t",method,":"
		for pct in pct_features_list:
			num_features = int(pct * len(array[0]))
			start = time()
			Y = decomposition.KernelPCA(kernel=method, n_components=num_features).fit_transform(array)
			end = time()
			print num_features, "\t", (end - start)

def RandomizedPCA(array, percent_samples):
	print "Randomized PCA", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = decomposition.RandomizedPCA(n_components=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def SparsePCA(array, percent_samples):
	print "Sparse PCA", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = decomposition.SparsePCA(n_components=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

##Confused about this
def GPFA(array, percent_samples):
	print "Gaussian Process Factor Analysis", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = decomposition.FactorAnalysis(n_components=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def NonGaussianICA(array, percent_samples):
	print "NonGaussian Independent Component Analysis", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = decomposition.FastICA(n_components=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def TruncatedSVD(array, percent_samples):
	print "Truncated Singular Value Decomposition", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = decomposition.TruncatedSVD(n_components=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def GaussianRandomProjection(array, percent_samples):
	print "Gaussian Random Projection", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = random_projection.GaussianRandomProjection(n_components=num_features).fit(array)
		end = time()
		print num_features, "\t", (end - start)

def SparseRandomProjection(array, percent_samples):
	print "Sparse Random Projection", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = random_projection.SparseRandomProjection().fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)

def UnivariateFeatureSelectionANOVA(array, percent_samples):
	print "Univariate Feature Selection with ANOVA", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		'''transform = feature_selection.SelectPercentile(feature_selection.f_classif)
		clf = pipeline.Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])
		Y = clf.set_params(anova__percentile=60, svc__C=.1).fit_transform(array)
		'''
		#f_regression = mem.cache(feature_selection.f_regression)  # caching function
		f_regression = feature_selection.f_regression
		anova = feature_selection.SelectPercentile(feature_selection.f_classif)
		clf = pipeline.Pipeline([('anova', anova)])

		clf.set_params(anova__percentile = pct)
		# Select the optimal percentage of features with grid search
		clf.fit_transform(array)  # set the best parameters
		end = time()
		print num_features, "\t", (end - start)

def FeatureAgglomeration(array, percent_samples):
	print "Feature Agglomeration", percent_samples*100, "% of training data."
	print "Features\tTime" 

	array = array[:int(percent_samples * len(array))]
	for pct in pct_features_list:
		num_features = int(pct * len(array[0]))
		start = time()
		Y = cluster.FeatureAgglomeration(n_clusters=num_features).fit_transform(array)
		end = time()
		print num_features, "\t", (end - start)


def main():
	array = import_data()
	test_labels = numpy.random.randint(2, size=len(array)).tolist()

	#LDA(array, test_labels)
	#Isomap(array, .02)
	#LLE(array, .02)
	#ModifiedLLE(array, .02)
	#Hessian(array, .02)
	#SpectralEmbedding(array, .02)
	#LTSA(array, .02)
	#MDSMetric(array, .02)
	#MDSNonmetric(array, .02)
	#TSNE(array, .02)

	#PCA(array, .02)
	#KernelPCA(array, .02)
	#RandomizedPCA(array, .02)
	#SparsePCA(array, .02)
	#GPFA(array, .02)
	#NonGaussianICA(array, .02)
	#TruncatedSVD(array, .02)
	#GaussianRandomProjection(array, .02)
	#SparseRandomProjection(array, .02)
	#UnivariateFeatureSelectionANOVA(array, .02)
	FeatureAgglomeration(array, .02)
		
main()