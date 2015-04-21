package darks.learning.classifier.maxent;

import java.io.OutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.corpus.Documents;
import darks.learning.corpus.Documents.Document;

/**
 * Train maxent model by GIS
 * 
 * @author  Darks.Liu
 *
 */
public class GISMaxent extends Maxent
{
    
    private static final Logger log = LoggerFactory.getLogger(GISMaxent.class);
    
    private static final double DEFAULT_MIN_ERROR = 0.0001;
    
    DoubleMatrix empiricalE = null;
    
    DoubleMatrix modelE = null;
    
    DoubleMatrix lambda = null;
    
    DoubleMatrix lastLambda = null;
    
    Map<FeaturePair, Integer> featureIndexMap = null;
    
    Map<FeaturePair, Long> featureMap = null;
    
	HashMap<String, Integer> termIndexMap = new HashMap<String, Integer>();
    
    int featureIndexSeed = 0;
    
    int maxFeatureCount = 0;
    
    Map<String, Integer> labelIndexMap = null;
    
    double minError = DEFAULT_MIN_ERROR;
    
    int[][] ctxIndexs;
    
    int[][] modelIndexs;
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void train(Documents docs, int maxIteration)
    {
    	long start = System.currentTimeMillis();
        initParam(docs);
        computeEmpiricalExpect(docs);
        for (int i = 0; i < maxIteration; i++)
        {
            if (nextIteration(docs))
                break;
            if (checkConverge(i))
                break;
        }
        long cost = System.currentTimeMillis() - start;
        log.info("Cost:" + cost);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public int predict(String[] input)
    {
    	int[] termIndexs = new int[input.length];
    	for (int i = 0; i < input.length; i++)
    	{
    		String term = input[i];
    		Integer index = termIndexMap.get(term);
    		int v = index == null ? -1 : index;
    		termIndexs[i] = v;
    	}
        DoubleMatrix probYX = computeProbYX(termIndexs);
        return SimpleBlas.iamax(probYX);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean saveModel(OutputStream out)
    {
        return false;
    }
    
    
    private void initParam(Documents docs)
    {
    	int termIndexSeed = 0;
        featureIndexMap = new HashMap<FeaturePair, Integer>();
        featureMap = new HashMap<FeaturePair, Long>();
        labelIndexMap = new HashMap<String, Integer>();
        ctxIndexs = new int[(int)docs.getDocsCount()][];
        int docIndex = 0;
        for (Entry<String, List<Document>> entry : docs.getLabelsMap().entrySet())
        {
            String label = entry.getKey();
            if (!labelIndexMap.containsKey(label))
            {
                int labelIndex = labels.size();
                labelIndexMap.put(label, labelIndex);
                labels.add(label);
            }
            for (Document doc : entry.getValue())
            {
                int termIndex = 0;
                int[] indexs = new int[doc.getTerms().size()];
                for (String term : doc.getTerms())
                {
                    FeaturePair pair = new FeaturePair(label, term);
                    Integer index = featureIndexMap.get(pair);
                    if (index == null)
                        index = featureIndexSeed++;
                    featureIndexMap.put(pair, index);
                    
                    index = termIndexMap.get(term);
                    if (index == null){
                        index = termIndexSeed++;
                        termIndexMap.put(term, index);
                    }
                    indexs[termIndex++] = index;
                    
                    Long count = featureMap.get(pair);
                    if (count == null)
                        count = 0l;
                    featureMap.put(pair, ++count);
                }
                ctxIndexs[docIndex++] = indexs;
                maxFeatureCount = Math.max(maxFeatureCount, doc.getTerms().size());
            }
        }
        modelIndexs = new int[termIndexMap.size()][labelIndexMap.size()];
        for (Entry<FeaturePair, Integer> featureIndex : featureIndexMap.entrySet())
        {
        	FeaturePair pair = featureIndex.getKey();
        	Integer index = featureIndex.getValue();
        	int labelIndex = labelIndexMap.get(pair.label);
        	int termIndex = termIndexMap.get(pair.term);
        	modelIndexs[termIndex][labelIndex] = index;
        }
        lambda = DoubleMatrix.zeros(featureIndexMap.size());
        empiricalE = DoubleMatrix.zeros(featureIndexMap.size());
        modelE = DoubleMatrix.zeros(featureIndexMap.size());
        log.info("Feature size " + modelE.length + "/" + termIndexMap.size() + "*" + labelIndexMap.size());
    }
    
    private void computeEmpiricalExpect(Documents docs)
    {
        long docsSize = docs.getDocsCount();
        for (Entry<FeaturePair, Long> entry : featureMap.entrySet())
        {
            FeaturePair pair = entry.getKey();
            long count = entry.getValue() == null ? 0 : entry.getValue();
            Integer index = featureIndexMap.get(pair);
            if (index == null)
                continue;
            empiricalE.put(index, (double) count / (double) docsSize);
        }
    }
    
    private boolean nextIteration(Documents docs)
    {
        computeModelExpect(docs);
        lastLambda = lambda.dup();
        for (int i = 0; i < modelE.length; i++)
        {
            double v = 1.0 / maxFeatureCount * Math.log(empiricalE.get(i) / modelE.get(i));
            lambda.put(i, lambda.get(i) + v);
        }
        return false;
    }
    
    private void computeModelExpect(Documents docs)
    {
    	modelE = DoubleMatrix.zeros(modelE.length);
        int labelSize = labels.size();
        long docsSize = docs.getDocsCount();
        for (int[] termIndexs : ctxIndexs)
        {
            DoubleMatrix probYX = computeProbYX(termIndexs);
            for (int j = 0; j < termIndexs.length; j++)
            {
                for (int i = 0; i < labelSize; i++)
                {
                    int index = modelIndexs[termIndexs[j]][i];
                    if (index >= 0)
                    	modelE.put(index, modelE.get(index) + probYX.get(i) / (double) docsSize);
                }
            }
        }
    }
    
    //P(Y|X)
    private DoubleMatrix computeProbYX(int[] termIndexs)
    {
        double sum = 0;
        int labelSize = labels.size();
        DoubleMatrix prob = DoubleMatrix.zeros(labelSize);
        for (int i = 0; i < labelSize; i++)
        {
            double lambdaSum = 0;
            for (int j = 0; j < termIndexs.length; j++)
            {
                int index = modelIndexs[termIndexs[j]][i];
                if (index >= 0)
                	lambdaSum += lambda.get(index);
            }
            lambdaSum = Math.exp(lambdaSum);
            prob.put(i, lambdaSum);
            sum += lambdaSum;
        }
        return prob.divi(sum);
    }
    
    private boolean checkConverge(int epuchNum)
    {
        DoubleMatrix mean = MatrixFunctions.abs(lastLambda.sub(lambda));
        int maxIndex = SimpleBlas.iamax(mean);
        if (mean.get(maxIndex) >= minError)
        {
            log.info("GIS iteration " + epuchNum + " error:" + mean.get(maxIndex));
            return false;
        }
        log.info("GIS converge on " + mean.get(maxIndex));
        return true;
    }
    
    class FeaturePair
    {
        String label;
        
        String term;

        public FeaturePair()
        {
        }

        public FeaturePair(String label, String term)
        {
            super();
            this.label = label;
            this.term = term;
        }

        public void set(String label, String term)
        {
            this.label = label;
            this.term = term;
        }

        @Override
        public int hashCode()
        {
            final int prime = 31;
            int result = 1;
            result = prime * result + getOuterType().hashCode();
            result = prime * result + ((label == null) ? 0 : label.hashCode());
            result = prime * result + ((term == null) ? 0 : term.hashCode());
            return result;
        }

        @Override
        public boolean equals(Object obj)
        {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            FeaturePair other = (FeaturePair)obj;
            if (!getOuterType().equals(other.getOuterType()))
                return false;
            if (label == null)
            {
                if (other.label != null)
                    return false;
            }
            else if (!label.equals(other.label))
                return false;
            if (term == null)
            {
                if (other.term != null)
                    return false;
            }
            else if (!term.equals(other.term))
                return false;
            return true;
        }

        private GISMaxent getOuterType()
        {
            return GISMaxent.this;
        }
        
    }
}
