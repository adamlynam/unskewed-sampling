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
 *    SMOTE.java
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.meta;

import weka.classifiers.RandomizableSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.*;
import weka.core.AdditionalMeasureProducer;

/**
 <!-- globalinfo-start -->
 * Class for bagging a classifier to reduce variance. Can do classification and regression depending on the base learner. <br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Leo Breiman (1996). SMOTE predictors. Machine Learning. 24(2):123-140.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman1996,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {2},
 *    pages = {123-140},
 *    title = {UnskewedSampling predictors},
 *    volume = {24},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)</pre>
 * 
 * <pre> -O
 *  Calculate the out of bag error.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.REPTree)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.REPTree:
 * </pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf (default 2).</pre>
 * 
 * <pre> -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).</pre>
 * 
 * <pre> -N &lt;number of folds&gt;
 *  Number of folds for reduced error pruning (default 3).</pre>
 * 
 * <pre> -S &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 * <pre> -P
 *  No pruning.</pre>
 * 
 * <pre> -L
 *  Maximum tree depth (default -1, no maximum)</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Bernhard Pfahringer (bernhard@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $
 */
public class UnskewedSampling
  extends RandomizableSingleClassifierEnhancer 
  implements WeightedInstancesHandler, 
             TechnicalInformationHandler, AdditionalMeasureProducer {

  /** for serialization */
  static final long serialVersionUID = -505879962237199703L;
  
  /** The number of minority examples to draw, as a percentage of the number of minority examples */
  protected double m_MinoritySampleSize = 1.0;
  
  /** The number of majority examples to draw, as a percentage of the number of majority examples */
  protected double m_MajoritySampleSize = 0.5;
    
  /**
   * Constructor.
   */
  public UnskewedSampling()
  {
    m_Classifier = new weka.classifiers.trees.J48();
  }
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
 
    return "Class for sampling differently from the minority and majority examples to reduce skew. Can do classification "
      + "and regression depending on the base learner. \n\n"
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Leo Breiman");
    result.setValue(Field.YEAR, "1996");
    result.setValue(Field.TITLE, "UnderBag predictors");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "24");
    result.setValue(Field.NUMBER, "2");
    result.setValue(Field.PAGES, "123-140");
    
    return result;
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  protected String defaultClassifierString() {
    
    return "weka.classifiers.trees.REPTree";
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(2);

    newVector.addElement(new Option(
              "\tThe number of minority examples to draw,\n" 
              + "\tas a percentage of the number of minority examples (default 1.0)",
              "m", 1, "-m"));
    newVector.addElement(new Option(
              "\tThe number of majority examples to draw,\n" 
              + "\tas a percentage of the number of majority examples (default 0.5)",
              "M", 1, "-M"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    return newVector.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -P
   *  Size of each bag, as a percentage of the
   *  training set size. (default 100)</pre>
   * 
   * <pre> -O
   *  Calculate the out of bag error.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.REPTree)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.REPTree:
   * </pre>
   * 
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf (default 2).</pre>
   * 
   * <pre> -V &lt;minimum variance for split&gt;
   *  Set minimum numeric class variance proportion
   *  of train variance for split (default 1e-3).</pre>
   * 
   * <pre> -N &lt;number of folds&gt;
   *  Number of folds for reduced error pruning (default 3).</pre>
   * 
   * <pre> -S &lt;seed&gt;
   *  Seed for random data shuffling (default 1).</pre>
   * 
   * <pre> -P
   *  No pruning.</pre>
   * 
   * <pre> -L
   *  Maximum tree depth (default -1, no maximum)</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    String minoritySampleSize = Utils.getOption('m', options);
    if (minoritySampleSize.length() != 0)
    {
      setMinoritySampleSize(Double.parseDouble(minoritySampleSize));
    } else
    {
      setMinoritySampleSize(1.0);
    }
    
    String majoiritySampleSize = Utils.getOption('M', options);
    if (majoiritySampleSize.length() != 0)
    {
      setMajoritySampleSize(Double.parseDouble(majoiritySampleSize));
    } else
    {
      setMajoritySampleSize(0.5);
    }

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions()
  {
    String [] superOptions = super.getOptions();
    String [] options = new String [superOptions.length + 4];

    int current = 0;
    options[current++] = "-m";
    options[current++] = "" + getMinoritySampleSize();
    options[current++] = "-M";
    options[current++] = "" + getMajoritySampleSize();

    System.arraycopy(superOptions, 0, options, current, 
		     superOptions.length);

    current += superOptions.length;
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String minoritySampleSizeTipText()
  {
    return "The number of minority examples to draw, as a percentage of the number of minority examples.";
  }

  /**
   * Gets the number of minority examples, as a percentage of the number of minority examples.
   *
   * @return the minority sample size, as a percentage.
   */
  public double getMinoritySampleSize()
  {
    return m_MinoritySampleSize;
  }
  
  /**
   * Sets the number of minority examples, as a percentage of the number of minority examples.
   *
   * @param newMinoritySampleSize the minority sample size, as a percentage.
   */
  public void setMinoritySampleSize(double newMinoritySampleSize)
  {
    m_MinoritySampleSize = newMinoritySampleSize;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String majoritySampleSizeTipText()
  {
    return "The number of majority examples to draw, as a percentage of the number of majority examples.";
  }

  /**
   * Gets the number of majority examples, as a percentage of the number of majority examples.
   *
   * @return the majority sample size, as a percentage.
   */
  public double getMajoritySampleSize()
  {
    return m_MajoritySampleSize;
  }
  
  /**
   * Sets the number of majority examples, as a percentage of the number of majority examples.
   *
   * @param newMajoritySampleSize the majority sample size, as a percentage.
   */
  public void setMajoritySampleSize(double newMajoritySampleSize)
  {
    m_MajoritySampleSize = newMajoritySampleSize;
  }
  
  /**
   * UnskewedSampling method.
   *
   * @param data the training data to be used for generating the
   * bagged classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    List<Instance> minorityClassInstances = new ArrayList<Instance>();
    List<Instance> majorityClassInstances = new ArrayList<Instance>();
    for(int i = 0; i < data.numInstances(); i++)
    {
      Instance instance = data.instance(i);
      if (instance.classValue() > 0.0)
      {
	majorityClassInstances.add(instance);
      } else
      {
	minorityClassInstances.add(instance);
      }
    }

    if (minorityClassInstances.size() > majorityClassInstances.size())
    {
      // swap to make minorityClassInstances the minority class
      List<Instance> temp = minorityClassInstances;
      minorityClassInstances = majorityClassInstances;
      majorityClassInstances = temp;
    }

    int minorityClassGoal = (int)Math.round(minorityClassInstances.size() * m_MinoritySampleSize); // number of minority class examples to draw
    int majorityClassGoal = (int)Math.round(majorityClassInstances.size() * m_MajoritySampleSize); // number of majority class examples to draw
    
    Random random = new Random(m_Seed);
    
    Instances unskewedData = new Instances(data, 0);
    for(int i = 0; i < minorityClassGoal; i++)
    {
      int nextMinorityInstance = random.nextInt(minorityClassInstances.size());
      unskewedData.add(minorityClassInstances.get(nextMinorityInstance));
    }
    for(int i = 0; i < majorityClassGoal; i++)
    {
      int nextMajorityInstance = random.nextInt(majorityClassInstances.size());
      unskewedData.add(majorityClassInstances.get(nextMajorityInstance));
    }

    if (m_Classifier instanceof Randomizable)
    {
        ((Randomizable) m_Classifier).setSeed(random.nextInt());
    }

    // build the classifier
    m_Classifier.buildClassifier(unskewedData);
  }

  /**
   * Calculates the class membership probabilities for the given test
   * instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @throws Exception if distribution can't be computed successfully 
   */
  public double[] distributionForInstance(Instance instance) throws Exception {

    double [] sums = new double [instance.numClasses()], newProbs; 
    
    if (instance.classAttribute().isNumeric() == true)
    {
        sums[0] += m_Classifier.classifyInstance(instance);
    }
    else
    {
        newProbs = m_Classifier.distributionForInstance(instance);
        for (int j = 0; j < newProbs.length; j++)
        sums[j] += newProbs[j];
    }
    if (instance.classAttribute().isNumeric() == true || Utils.eq(Utils.sum(sums), 0))
    {
      return sums;
    }
    else
    {
      Utils.normalize(sums);
      return sums;
    }
  }
  
  
  
    public Enumeration enumerateMeasures()
    {
        if (m_Classifier instanceof AdditionalMeasureProducer)
            return ((AdditionalMeasureProducer)m_Classifier).enumerateMeasures();
        else
            return new Vector(0).elements();
    }

    public double getMeasure(String additionalMeasureName)
    {
        if (m_Classifier instanceof AdditionalMeasureProducer)
            return ((AdditionalMeasureProducer)m_Classifier).getMeasure(additionalMeasureName);
        else
            throw new IllegalArgumentException("Additional measures not supported by base classifier.");
    }

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  public String toString() {
    
    if (m_Classifier == null)
    {
      return "UnskewedSampling: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("All the base classifiers: \n\n");
    text.append(m_Classifier.toString() + "\n\n");
    return text.toString();
  }

  public String getRevision() {
    //return RevisionUtils.extract("$Revision: 1.41 $");
    return "1.0";
  }
  
  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
    runClassifier(new UnskewedSampling(), argv);
  }
  
  private int roughlyBalancedBaggingMajorityClassCountCalculation(int minorityClassGoal, double minorityClassChance, int randomSeed)
  {
      // check the minority chance is within the 0->1 range, if it is not then return the minority goal
      if (minorityClassChance <= 0.0 || minorityClassChance > 1.0)
          return minorityClassGoal;
      
      int minorityCount = 0;
      int majorityCount = 0;
      
      Random random = new Random(randomSeed);
      
      while (minorityCount < minorityClassGoal)
      {
          if (random.nextDouble() < minorityClassChance)
              minorityCount++;
          else
              majorityCount++;
      }
      
      return majorityCount;
  }
}
