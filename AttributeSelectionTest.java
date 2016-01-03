import weka.attributeSelection.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.*;
import weka.classifiers.*;
import weka.classifiers.meta.*;
import weka.classifiers.trees.*;
import weka.filters.*;

import java.util.*;

/**
 * performs attribute selection using ClassifierSubsetEval and GreedyStepwise
 * (backwards) and trains J48 with that. Needs 3.5.5 or higher to compile.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class AttributeSelectionTest {

  /**
   * uses the meta-classifier
   */
  protected static void useClassifier(Instances data) throws Exception {
    System.out.println("\n1. Meta-classfier");
    AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
    ClassifierSubsetEval eval = new ClassifierSubsetEval();
    BestFirst search = new BestFirst();
    //search.setSearchBackwards(true);
    search.setStartSet("1-175");
    search.setSearchTermination(5);
    J48 base = new J48();
    classifier.setClassifier(base);
    classifier.setEvaluator(eval);
    classifier.setSearch(search);
    System.out.println("init eval");
    Evaluation evaluation = new Evaluation(data);
    evaluation.crossValidateModel(classifier, data, 10, new Random(1));
    System.out.println(evaluation.toSummaryString());
  }

  /**
   * uses the filter
   */
  protected static void useFilter(Instances data) throws Exception {
    System.out.println("\n2. Filter");
    weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
    ClassifierSubsetEval eval = new ClassifierSubsetEval();
    BestFirst search = new BestFirst();
    //search.setSearchBackwards(true);
    search.setStartSet("1-175");
    search.setSearchTermination(3);
    filter.setEvaluator(eval);
    filter.setSearch(search);
    filter.setInputFormat(data);
    Instances newData = Filter.useFilter(data, filter);
    //System.out.println(newData);
  }

  /**
   * uses the low level approach
   */
  protected static int[] useLowLevel(Instances data, ASSearch search, ASEvaluation eval) throws Exception {
    System.out.println("\n3. Low-level");
    AttributeSelection attsel = new AttributeSelection();
    //ClassifierSubsetEval eval = new ClassifierSubsetEval();
    //BestFirst search = new BestFirst();
    attsel.setEvaluator(eval);
    attsel.setSearch(search);
    attsel.SelectAttributes(data);
    int[] indices = attsel.selectedAttributes();
    System.out.println("selected attribute indices (starting with 0):\n" + Utils.arrayToString(indices));
    //System.out.println("---\n" + attsel.toResultsString());

    return indices;
  }

  /**
   * takes a dataset as first argument
   *
   * @param args        the commandline arguments
   * @throws Exception  if something goes wrong
   */
  public static void main(String[] args) throws Exception {
    // load data
    System.out.println("\n0. Loading data");
    DataSource source = new DataSource(args[0]);
    Instances data = source.getDataSet();
    if (data.classIndex() == -1)
      data.setClassIndex(data.numAttributes() - 1);

    // 1. meta-classifier
    //useClassifier(data);

    // 2. filter
    //useFilter(data);

    // 3. low-level

    //------------------FULL RANKERS---------------//

    Ranker ranker = new Ranker();

    System.out.println("GAINRATIO");
    GainRatioAttributeEval gainratio = new GainRatioAttributeEval();
    //useLowLevel(data, ranker, gainratio);

    System.out.println("INFOGAIN");
    InfoGainAttributeEval infoGain = new InfoGainAttributeEval();
    //useLowLevel(data, ranker, infoGain);

    System.out.println("CHISQUARED");
    ChiSquaredAttributeEval chi = new ChiSquaredAttributeEval();
    //useLowLevel(data, ranker, chi);

    System.out.println("ONER");
    OneRAttributeEval oner = new OneRAttributeEval();
    //useLowLevel(data, ranker, oner);

    System.out.println("RELIEFF");
    ReliefFAttributeEval relief = new ReliefFAttributeEval();
    //useLowLevel(data, ranker, relief);

    System.out.println("SYMMETRICALUNCERT");
    SymmetricalUncertAttributeEval sym = new SymmetricalUncertAttributeEval();
    //useLowLevel(data, ranker, sym);
    



    //-----------------SUBSETS --- BST FIRST --------------//

    //ClassifierSubsetEval eval = new ClassifierSubsetEval();
    //eval.setClassifier(new weka.classifiers.bayes.NaiveBayes());
    
    
    BestFirst bfs = new BestFirst();
     bfs.setDirection(new SelectedTag(0,BestFirst.TAGS_SELECTION));
    bfs.setSearchTermination(5);
    bfs.setStartSet("1-174");
   

    System.out.println("CFSSUBSET");
    CfsSubsetEval cfs = new CfsSubsetEval();
    //useLowLevel(data, bfs, cfs);

    System.out.println("CONSISTENCYSUBSET");
    ConsistencySubsetEval consistent = new ConsistencySubsetEval();
    //useLowLevel(data, bfs, consistent);

    System.out.println("FILTEREDSUBSET");
    FilteredSubsetEval filtered = new FilteredSubsetEval();
    //useLowLevel(data, bfs, filtered);

    System.out.println("WRAPPERSUBSET");
    WrapperSubsetEval wrapper = new WrapperSubsetEval();
    //useLowLevel(data, bfs , wrapper);


//--------------------SUBSETS----GREEDYSSTEPWISE--------------//
    GreedyStepwise greedy = new GreedyStepwise();
    greedy.setNumToSelect(10);

    System.out.println("CFSSUBSET");
    useLowLevel(data, greedy, cfs);

    System.out.println("CONSISTENCYSUBSET");
    useLowLevel(data, greedy, consistent);

    System.out.println("FILTEREDSUBSET");
    useLowLevel(data, greedy, filtered);

    System.out.println("WRAPPERSUBSET");
    useLowLevel(data, greedy , wrapper);

//--------------------SUBSETS GENETIC --------------------//

    System.out.println("----GENETIC----");

    GeneticSearch genetic = new GeneticSearch();
    
    System.out.println("CFSSUBSET");
    useLowLevel(data, genetic, cfs);

    System.out.println("CONSISTENCYSUBSET");
    useLowLevel(data, genetic, consistent);

    System.out.println("FILTEREDSUBSET");
    useLowLevel(data, genetic, filtered);

    System.out.println("WRAPPERSUBSET");
    useLowLevel(data, genetic, wrapper);

// ------------SUBSETS RANDOMSEARCH ----------------//

    System.out.println("----RANDOM----");

    RandomSearch random = new RandomSearch();
    random.setVerbose(true);
    
    System.out.println("CFSSUBSET");
    useLowLevel(data, random, cfs);

    System.out.println("CONSISTENCYSUBSET");
    useLowLevel(data, random, consistent);

    System.out.println("FILTEREDSUBSET");
    useLowLevel(data, random, filtered);

    System.out.println("WRAPPERSUBSET");
    useLowLevel(data, random, wrapper);

  }
}