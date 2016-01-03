/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    WekaClassify.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *
 */

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.core.converters.ConverterUtils.DataSource;


import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Vector;
import  java.text.*;

/**
 *  Java program for using WEKA Classifications
 *  It was originally started with a demo program from the WEKA examples.
 * Check out the Evaluation class for more details.
 *
 * @author Mayank Mahajan
 */

public class WekaClassify {
    /** the classifier used internally */
    protected Classifier m_Classifier = null;

    /** the filter to use */
    protected Filter m_Filter = null;

    /** the training file */
    protected String m_TrainingFile = null;

    /** the training instances */
    protected Instances m_Training = null;

    /** for evaluating the classifier */
    protected Evaluation m_Evaluation = null;

    /** evaluation type */
    protected String m_EvaluationMode = null;

    /** evaluation option */
    protected String[] m_EvaluationOptions = null;

    /**
     * initializes the demo
     */
    public WekaClassify() {
        super();
    }

    /**
     * sets the classifier to use
     * @param name        the classname of the classifier
     * @param options     the options for the classifier
     */
    public void setClassifier(String name, String[] options) throws Exception {
        m_Classifier = Classifier.forName(name, options);
    }

    public void setClassifierDirectly(Classifier c) {
        m_Classifier = c;
    }
    /**
     * sets the filter to use
     * @param name        the classname of the filter
     * @param options     the options for the filter
     */
    public void setFilter(String name, String[] options) throws Exception {
        m_Filter = (Filter) Class.forName(name).newInstance();
        if (m_Filter instanceof OptionHandler)
            ((OptionHandler) m_Filter).setOptions(options);
    }

    /**
     * sets the file to use for training
     */
    public void setTraining(String name) throws Exception {
        m_TrainingFile = name;
        m_Training     = new Instances(
                new DataSource(m_TrainingFile).getDataSet());
        m_Training.setClassIndex(m_Training.numAttributes() - 1);
    }

    public void setEvaluation(String evaluation, String[] options) throws  Exception {

        m_EvaluationMode = evaluation;

        if (options.length != 0)
        {
            m_EvaluationOptions = options;
        }

    }
    /**
     * runs 10fold CV over the training file
     */
    public void execute() throws Exception {
        // run filter
        if (m_Filter != null) {
            m_Filter.setInputFormat(m_Training);
            Instances filtered = Filter.useFilter(m_Training, m_Filter);

            // train classifier with filtered on complete file for tree
            RunClassify(filtered);
        } else
        {
            // train classifier on complete file for tree
            RunClassify(m_Training);
        }
    }

    public void RunClassify(Instances anInstance)  {

        try {
        m_Classifier.buildClassifier(anInstance);
        EvaluationRun(anInstance);
    }

        catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void EvaluationRun(Instances anInstance) throws Exception {

        m_Evaluation = new Evaluation(anInstance);
        // Evaluation modes are
        // cross-validation (assumed 10 fold)
        // training-data
        // split (optional percentage value, default 66%)
        if (m_EvaluationMode != null)
        {
            switch (m_EvaluationMode) {
                case "cross-validation":
                    int fold = 10; //default
                    if (m_EvaluationOptions.length > 0) {
                        fold = Integer.parseInt(m_EvaluationOptions[0]);
                    }
                    FoldRun(anInstance, fold);
                    break;
                case "training-set":
                    TrainingSetRun(anInstance);
                    break;
                case "split":
                    double split = 0.66;
                    if (m_EvaluationOptions.length > 0 ) {
                        split = Double.parseDouble(m_EvaluationOptions[0]);
                    }
                    SplitRun(anInstance, split);
                    break;
                default:
                    System.out.println("Incorrect Evaluation Mode: " + m_EvaluationMode);
                    System.exit(-1);
                    break;
            }
        } else
            TrainingSetRun(anInstance);

    }


    public void SplitRun(Instances anInstance, double split) throws  Exception {

        anInstance.randomize(new java.util.Random(0));

        int trainSize = (int) Math.round(anInstance.numInstances() * split);
        int testSize = anInstance.numInstances() - trainSize;
        Instances train = new Instances(anInstance, 0, trainSize);
        Instances test = new Instances(anInstance, trainSize, testSize);

        NumberFormat percentFormat = NumberFormat.getPercentInstance();
        percentFormat.setMaximumFractionDigits(1);
        String result = percentFormat.format(split);

        System.out.println("Evaluating with : Split percentage : "+ result);
        m_Evaluation.evaluateModel(m_Classifier, anInstance);
    }

    public void TrainingSetRun(Instances anInstance) throws  Exception {

        System.out.println("Evaluating with : Training Set : ");
        m_Evaluation.evaluateModel(m_Classifier, anInstance);
    }


    public void FoldRun(Instances anInstance, int fold) throws Exception {

        // 10fold CV with seed=1
        System.out.println("Evaluating with : Fold : "+ fold);

        m_Evaluation.crossValidateModel(
                m_Classifier, anInstance, fold, m_Training.getRandomNumberGenerator(1));

    }
    /**
     * outputs some data about the classifier
     */
    public String toString() {
        StringBuffer        result;

        result = new StringBuffer();
        result.append("Weka - Classify\n===========\n\n");

        result.append("Classifier...: "
                + m_Classifier.getClass().getName() + " "
                + Utils.joinOptions(m_Classifier.getOptions()) + "\n");
        if (m_Filter != null)
        {
            if (m_Filter instanceof OptionHandler)
            result.append("Filter.......: "
                    + m_Filter.getClass().getName() + " "
                    + Utils.joinOptions(((OptionHandler) m_Filter).getOptions()) + "\n");
        else
            result.append("Filter.......: "
                    + m_Filter.getClass().getName() + "\n");
        }
        result.append("Training file: "
                + m_TrainingFile + "\n");
        result.append("\n");

        result.append(m_Classifier.toString() + "\n");
        result.append(m_Evaluation.toSummaryString() + "\n");
        try {
            result.append(m_Evaluation.toMatrixString() + "\n");
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        try {
            result.append(m_Evaluation.toClassDetailsString() + "\n");
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        return result.toString();
    }

    public String toStringAbbrev() {
        StringBuffer result;
        result = new StringBuffer();
        result.append("Classifier...: "
                + m_Classifier.getClass().getName() + " "
                + Utils.joinOptions(m_Classifier.getOptions()) + "\n");
        result.append("Training file: " + m_TrainingFile + "\n");
        result.append("\n");
        result.append(m_Evaluation.toSummaryString() + "\n");
        try {
            result.append(m_Evaluation.toMatrixString() + "\n");
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        try {
            result.append(m_Evaluation.toClassDetailsString() + "\n");
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        return result.toString();

    } 


    public String getRawAccuracy() {
        StringBuffer result;
        String summary = m_Evaluation.toSummaryString();
    }

    /**
     * returns the usage of the class
     */
    public static String usage() {
        return
                "\nusage:\n  " + WekaClassify.class.getName()
                        + "  CLASSIFIER <classname> [options] \n"
                        + "  FILTER <classname> [options]\n"
                        + "  EVALUATION <evaluation-type> [options] \n"
                        + "  DATASET <trainingfile>\n\n"
                        + "e.g., \n"
                        + "  java -classpath \".:weka.jar\" WekaClassify \n"
                        + "    CLASSIFIER weka.classifiers.trees.J48 -U \n"
                        + "    FILTER weka.filters.unsupervised.instance.Randomize \n"
                        + "    EVALUATION cross-validation 10 \n"
                        + "    DATASET iris.arff\n";

    }

    /**
     * runs the program, the command line looks like this:<br/>
     * WekaClassify CLASSIFIER classname [options]
     *          FILTER classname [options]
     *          DATASET filename
     * <br/>
     * e.g., <br/>
     *   java -classpath ".:weka.jar" WekaClassify \<br/>
     *     CLASSIFIER weka.classifiers.trees.J48 -U \<br/>
     *     FILTER weka.filters.unsupervised.instance.Randomize \<br/>
     *     DATASET iris.arff<br/>
     */

    public static void main(String[] args) throws Exception {
        WekaClassify         wclass;

        // parse command line
        String classifier = "";
        String filter = "";
        String dataset = "";
        String evaluation = "";
        Vector classifierOptions = new Vector();
        Vector filterOptions = new Vector();
        Vector evaluationOptions = new Vector();

        int i = 0;
        String current = "";
        boolean newPart = false;
        do {
            // determine part of command line
            if (args[i].equals("CLASSIFIER")) {
                current = args[i];
                i++;
                newPart = true;
            }
            else if (args[i].equals("EVALUATION")) {
                current = args[i];
                i++;
                newPart = true;
            }
            else if (args[i].equals("FILTER")) {
                current = args[i];
                i++;
                newPart = true;
            }
            else if (args[i].equals("DATASET")) {
                current = args[i];
                i++;
                newPart = true;
            }

            if (current.equals("CLASSIFIER")) {
                if (newPart)
                    classifier = args[i];
                else
                    classifierOptions.add(args[i]);
            }
            if (current.equals("EVALUATION")) {
                if (newPart)
                    evaluation = args[i];
                else
                    evaluationOptions.add(args[i]);
            }
            else if (current.equals("FILTER")) {
                if (newPart)
                    filter = args[i];
                else
                    filterOptions.add(args[i]);
            }

            else if (current.equals("DATASET")) {
                if (newPart)
                    dataset = args[i];
            }

            // next parameter
            i++;
            newPart = false;
        }
        while (i < args.length);

        // minimum arguments provided?
        if ( classifier.equals("") || dataset.equals("") ) {
            System.out.println("Not all parameters provided!");
            System.out.println(WekaClassify.usage());
            System.exit(2);
        }

        // run
        wclass = new WekaClassify();

        wclass.setClassifier(
                classifier,
                (String[]) classifierOptions.toArray(new String[classifierOptions.size()]));

        if (!filter.equals(""))
            wclass.setFilter(filter, (String[]) filterOptions.toArray(new String[filterOptions.size()]));

        if (!evaluation.equals(""))
            wclass.setEvaluation(evaluation, (String[]) evaluationOptions.toArray(new String[evaluationOptions.size()]));

        wclass.setTraining(dataset);

        wclass.execute();

        System.out.println(wclass.toString());
    }
    
}
