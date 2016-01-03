import csv
import random
import numpy as np
from feast import *

percent_features = [
	0.1,
	0.9,
	]

dim_red_techniques = [
	"LDA",
	"Isomap",
	"LLE",
	"ModifiedLLE",
	"Hessian",
	"SpectralEmbedding,",
	"LTSA",
	"MDSMetric",
	"MDSNonmetric",
	"TSNE",
	"PCA",
	"KernelPCA",
	"RandomizedPCA",
	"SparsePCA",
	"GPFA",
	"NonGaussianICA",
	"TruncatedSVD",
	"GaussianRandomProjection",
	"SparseRandomProjection",
	"UnivariateFeatureSelectionANOVA"]

ranker_scoring_techniques = [
	"GainRatio",
	"InfoGain",
	"ChiSquared",
	"OneR",
	"ReliefF",
	"SymmetricalUncert"]

ranker_search_techniques = ["Ranker"]

subset_scoring_techniques = [
	"CfsSubset",
	"ConsistencySubset",
	"FilteredSubset",
	"WrapperSubset"]

subset_search_techniques = [
	"BestFirst",
	"GreedyStepwise",
	"GeneticSearch",
	"RandomSearch"]

classifiers = {
	"bayes":
		["weka.classifiers.bayes.BayesianLogisticRegression -D -Tl 5.0E-4 -S 0.5 -H 1 -V 0.27 -R R:0.01-316,3.16 -P 1 -F 2 -seed 1 -I 100 -N",
		"weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5",
		"weka.classifiers.bayes.DMNBtext -I 1",
		"weka.classifiers.bayes.NaiveBayes",
		"weka.classifiers.bayes.NaiveBayesUpdateable",
		],
	"functions":
		["weka.classifiers.functions.Logistic -R 1.0E-8 -M -1",
		#"weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a",
		"weka.classifiers.functions.RBFNetwork -B 2 -S 1 -R 1.0E-8 -M -1 -W 0.1",
		"weka.classifiers.functions.SimpleLogistic -I 0 -M 500 -H 50 -W 0.0",
		"weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\"",
		"weka.classifiers.functions.SPegasos -F 0 -L 1.0E-4 -E 500",
		"weka.classifiers.functions.VotedPerceptron -I 1 -E 1.0 -S 1 -M 10000",
		],
	"lazy":
		["weka.classifiers.lazy.IB1",
		"""weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"""",
		"weka.classifiers.lazy.KStar -B 20 -M a",
		"""weka.classifiers.lazy.LWL -U 0 -K -1 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" -W weka.classifiers.trees.DecisionStump""",
		],
	"meta":
		["weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump",
		"""weka.classifiers.meta.AttributeSelectedClassifier -E "weka.attributeSelection.CfsSubsetEval " -S "weka.attributeSelection.BestFirst -D 1 -N 5" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2""",
		"weka.classifiers.meta.Bagging -P 100 -S 1 -I 10 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.001 -N 3 -S 1 -L -1",
		"""weka.classifiers.meta.ClassificationViaClustering -W weka.clusterers.SimpleKMeans -- -N 2 -A "weka.core.EuclideanDistance -R first-last" -I 500 -S 10""",
		"weka.classifiers.meta.ClassificationViaRegression -W weka.classifiers.trees.M5P -- -M 4.0",
		"weka.classifiers.meta.CVParameterSelection -X 10 -S 1 -W weka.classifiers.rules.ZeroR",
		"""weka.classifiers.meta.Dagging -F 10 -S 1 -W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0" """,
		"weka.classifiers.meta.Decorate -E 15 -R 1.0 -S 1 -I 50 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2",
		"weka.classifiers.meta.END -S 1 -I 10 -W weka.classifiers.meta.nestedDichotomies.ND -- -S 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2",
		""" weka.classifiers.meta.FilteredClassifier -F "weka.filters.supervised.attribute.Discretize -R first-last" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2 """,
		""" weka.classifiers.meta.Grading -X 10 -M "weka.classifiers.rules.ZeroR " -S 1 -B "weka.classifiers.rules.ZeroR " """,
		"weka.classifiers.meta.LogitBoost -P 100 -F 0 -R 1 -L -1.7976931348623157E308 -H 1.0 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump",
		"weka.classifiers.meta.MultiBoostAB -C 3 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump",
		"weka.classifiers.meta.MultiClassClassifier -M 0 -R 2.0 -S 1 -W weka.classifiers.functions.Logistic -- -R 1.0E-8 -M -1",
		""" weka.classifiers.meta.MultiScheme -X 0 -S 1 -B "weka.classifiers.rules.ZeroR " """,
		"weka.classifiers.meta.nestedDichotomies.ClassBalancedND -S 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2",
		"weka.classifiers.meta.nestedDichotomies.DataNearBalancedND -S 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2",
		"weka.classifiers.meta.nestedDichotomies.ND -S 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2",
		"weka.classifiers.meta.OrdinalClassClassifier -W weka.classifiers.trees.J48 -- -C 0.25 -M 2",
		"weka.classifiers.meta.RacedIncrementalLogitBoost -C 500 -M 2000 -V 1000 -P 1 -S 1 -W weka.classifiers.trees.DecisionStump",
		"weka.classifiers.meta.RandomCommittee -S 1 -I 10 -W weka.classifiers.trees.RandomTree -- -K 0 -M 1.0 -S 1",
		"weka.classifiers.meta.RandomSubSpace -P 0.5 -S 1 -I 10 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.001 -N 3 -S 1 -L -1",
		""" weka.classifiers.meta.RotationForest -G 3 -H 3 -P 50 -F "weka.filters.unsupervised.attribute.PrincipalComponents -R 1.0 -A 5 -M -1" -S 1 -I 10 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2 """,
		""" weka.classifiers.meta.StackingC -X 10 -M "weka.classifiers.functions.LinearRegression -S 1 -R 1.0E-8" -S 1 -B "weka.classifiers.rules.ZeroR " """,
		"weka.classifiers.meta.ThresholdSelector -C 5 -X 3 -E 1 -R 0 -M FMEASURE -S 1 -W weka.classifiers.functions.Logistic -- -R 1.0E-8 -M -1",
		""" weka.classifiers.meta.Vote -S 1 -B "weka.classifiers.rules.ZeroR " -R AVG """,
		],
	"mi":
		[
		],
	"misc":
		["weka.classifiers.misc.HyperPipes",
		"weka.classifiers.misc.VFI -B 0.6"
		],
	"rules":
		["weka.classifiers.rules.ConjunctiveRule -N 3 -M 2.0 -P -1 -S 1",
		""" weka.classifiers.rules.DecisionTable -X 1 -S "weka.attributeSelection.BestFirst -D 1 -N 5" """,
		#"weka.classifiers.rules.DTNB -X 1",
		"weka.classifiers.rules.JRip -F 3 -N 2.0 -O 2 -S 1",
		"weka.classifiers.rules.NNge -G 5 -I 5",
		"weka.classifiers.rules.OneR -B 6",
		"weka.classifiers.rules.PART -M 2 -C 0.25 -Q 1",
		"weka.classifiers.rules.Ridor -F 3 -S 1 -N 2.0",
		"weka.classifiers.rules.ZeroR"
		],
	"trees":
		[
		"weka.classifiers.trees.ADTree -B 10 -E -3",
		#"weka.classifiers.trees.BFTree -S 1 -M 2 -N 5 -C 1.0 -P POSTPRUNED",
		"weka.classifiers.trees.DecisionStump",
		"weka.classifiers.trees.FT -I 15 -F 0 -M 15 -W 0.0",
		"weka.classifiers.trees.J48 -C 0.25 -M 2",
		"weka.classifiers.trees.J48graft -C 0.25 -M 2",
		"weka.classifiers.trees.LADTree -B 10",
		"weka.classifiers.trees.LMT -I -1 -M 15 -W 0.0",
		"weka.classifiers.trees.NBTree",
		"weka.classifiers.trees.RandomForest -I 100 -K 0 -S 1",
		"weka.classifiers.trees.RandomTree -K 0 -M 1.0 -S 1",
		"weka.classifiers.trees.REPTree -M 2 -V 0.001 -N 3 -S 1 -L -1",
		"weka.classifiers.trees.SimpleCart -S 1 -M 2.0 -N 5 -C 1.0",
		]
}

evaluation_techniques = [
	"10-Fold Cross validation",
	"Split",
	"Training Set"
]

#Search and scorer

def build_attribute_selection_technique_list():
	commands = []
	for percent in percent_features:
		for reduction_technique in dim_red_techniques:
			commands.append("PERCENT: " + str(percent) + " REDUCTION: " + reduction_technique)
		for ranker_search in ranker_search_techniques:
			for ranker_scorer in ranker_scoring_techniques:
				commands.append("PERCENT: " + str(percent) + " RANKER SCORER: " + ranker_scorer + 
					" RANKER SEARCHER: " + ranker_search)
		for subset_search in subset_search_techniques:
			for subset_scorer in subset_scoring_techniques:
				commands.append("PERCENT: " + str(percent) + " SUBSET SCORER: " + subset_scorer + 
					" SUBSET SEARCHER: " + subset_search)
	return commands


def build_full_commands():
	partial_commands = build_attribute_selection_technique_list()
	full_commands = []
	for command in partial_commands:
		for classifier_type in classifiers:
			for classifier in classifiers[classifier_type]:
				for evaluation_technique in evaluation_techniques:
					full_commands.append(command + " CLASSIFIER: " + classifier + " EVALUATION METHOD: " + evaluation_technique)
	return full_commands

def main():
	commands = build_full_commands()
	with open("command_list.txt", "wb") as outfile:
		for x in commands:
			outfile.write(x + "\n")
	print str(len(percent_features) * \
	(len(dim_red_techniques) + len(ranker_scoring_techniques) * len(ranker_search_techniques) + len(subset_scoring_techniques) * len(subset_search_techniques)) \
	* sum(len(classifiers[classifier_type]) for classifier_type in classifiers) \
	* len(evaluation_techniques)) + " commands created in command_list.txt"

main()