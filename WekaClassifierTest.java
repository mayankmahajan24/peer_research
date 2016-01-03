import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;




import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Vector;
import java.util.LinkedList;
import java.text.*;

/**
 *  Java program for using WEKA Classifications
 *  It was originally started with a demo program from the WEKA examples.
 * Check out the Evaluation class for more details.
 *
 * @author Mayank Mahajan
 */

public class WekaClassifierTest {

	static LinkedList<Classifier> classifiers = new LinkedList<Classifier>();
	final static String dataset = "all_combinedSMALL.csv";
	//final static String outfile = "outfile.csv";

	static WekaClassify wc = new WekaClassify();

	public static void main(String[] args) {
		setupClassifierArray(classifiers);

		try {
			wc.setTraining(dataset);

			for (Classifier c : classifiers) {

				wc.setClassifierDirectly(c);
				wc.execute();
				System.out.println(wc.toStringAbbrev());
			}
		}

		catch (Exception e) {
			e.printStackTrace();
		}
	}


	public static void setupClassifierArray(LinkedList<Classifier> m_classifiers) {
		try {
			m_classifiers.add(Classifier.forName("weka.classifiers.trees.J48", new String[] {"-U"}));
			m_classifiers.add(Classifier.forName("weka.classifiers.bayes.BayesianLogisticRegression", 
				new String[] {"-D", "-Tl", "5.0E-4", "-H", "1", "-V", "0.27", "-R", "R:0.01-316,3.16",
				"-P", "1", "-F", "2", "-seed", "1", "-I", "100", "-N"}));
			m_classifiers.add(Classifier.forName("weka.classifiers.bayes.BayesNet", 
				new String[] {"-D", "-Q", "weka.classifiers.bayes.net.search.local.K2", "--", "-P", "1", 
				"-S", "BAYES", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A", "0.5"}));
			m_classifiers.add(Classifier.forName("weka.classifiers.bayes.ComplementNaiveBayes", new String[] {"-S", "1.0"}));
			m_classifiers.add(Classifier.forName("weka.classifiers.bayes.DMNBtext", new String[] {"-I", "1"}));
			m_classifiers.add(Classifier.forName("weka.classifiers.bayes.NaiveBayes", new String[] {}));
			//Won't work m_classifiers.add(Classifier.forName("weka.classifiers.bayes.NaiveBayesMultinomial", new String[] {}));
			//Won't work NaiveBayesMultinomialUpdateable
			//Won't work NaiveBayesSimple (BRF1: s.d. is 0 for class positive)
			m_classifiers.add(Classifier.forName("weka.classifiers.bayes.NaiveBayesUpdateable", new String[] {}));

		}

		catch (Exception e) {
			e.printStackTrace();
		}

	}

}