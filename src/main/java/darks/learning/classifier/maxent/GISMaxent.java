/**
 * 
 * Copyright 2014 The Darks Learning Project (Liu lihua)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package darks.learning.classifier.maxent;

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
	
	public static final String REPORT_EPOCH_NUM = "epuchNum";
	
	public static final String REPORT_LIKELIDHOOD = "likelihood";
	
	public static final String REPORT_ERROR = "error";
    
    private static final Logger log = LoggerFactory.getLogger(GISMaxent.class);
    
    private static final double DEFAULT_MIN_ERROR = 0.0001;
    
    /**
     * empirical expected vector
     */
    DoubleMatrix empiricalE = null;

    /**
     * model expected vector
     */
    DoubleMatrix modelE = null;
    
    DoubleMatrix lambda = null;
    
    DoubleMatrix lastLambda = null;
    
    /**
     * Store <<term,label>,index>, index is the model expect index.
     */
    Map<FeaturePair, Integer> featureIndexMap = null;

    /**
     * Store <<term,label>,count>
     */
    Map<FeaturePair, Long> featureMap = null;
    
    /**
     * Store the index of terms in vector
     */
    Map<String, Integer> termIndexMap = null;
    
    /**
     * Maximum feature/term count as C
     */
    int maxFeatureCount = 0;
    
    /**
     * Store the index of labels
     */
    Map<String, Integer> labelIndexMap = null;
    
    double minError = DEFAULT_MIN_ERROR;
    
    /**
     * Array of documents' label indexs
     */
    int[] ctxLabelIndexs;
    
    /**
     * [document index][local term index]=[global term index]
     */
    int[][] ctxIndexs;
    
    /**
     * [global term index][label index]=[model expect index]
     */
    int[][] modelIndexs;
    
    double loglikelihood;
    
    public GISMaxent()
    {
        
    }
    
    public GISMaxent(GISModel model)
    {
        loadModel(model);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public MaxentModel train(Documents docs, int maxIteration)
    {
    	log.info("Start to train GIS maxent model.");
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
        releaseMemory();
        long cost = System.currentTimeMillis() - start;
        log.info("Complete to train GIS model. Cost:" + cost);
        return new GISModel(labels, modelIndexs, termIndexMap, lambda);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public DoubleMatrix predictMatrix(String[] input)
    {
    	int[] termIndexs = new int[input.length];
    	for (int i = 0; i < input.length; i++)
    	{
    		String term = input[i];
    		Integer index = termIndexMap.get(term);
    		int v = index == null ? -1 : index;
    		termIndexs[i] = v;
    	}
        return computeProbYX(termIndexs);
    }
    
    private void releaseMemory()
    {
    	ctxLabelIndexs = null;
        ctxIndexs = null;
        empiricalE = null;
        modelE = null;
        lastLambda = null;
        featureMap = null;
    }
    
    
    private void initParam(Documents docs)
    {
        int featureIndexSeed = 0;
    	int termIndexSeed = 0;
    	termIndexMap = new HashMap<String, Integer>();
        featureIndexMap = new HashMap<FeaturePair, Integer>();
        featureMap = new HashMap<FeaturePair, Long>();
        labelIndexMap = new HashMap<String, Integer>();
        ctxIndexs = new int[(int)docs.getDocsCount()][];
        ctxLabelIndexs = new int[(int)docs.getDocsCount()];
        int docIndex = 0;
        for (Entry<String, List<Document>> entry : docs.getLabelsMap().entrySet())
        {
            String label = entry.getKey();
            Integer labelIndex = labelIndexMap.get(label);
            if (labelIndex == null)
            {
                labelIndex = labels.size();
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
                ctxLabelIndexs[docIndex] = labelIndex;
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
        log.info("GIS document count " + docs.getDocsCount());
        log.info("GIS feature size " + modelE.length + "/" + termIndexMap.size() + "*" + labelIndexMap.size());
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
    	loglikelihood = 0;
    	modelE = DoubleMatrix.zeros(modelE.length);
        int labelSize = labels.size();
        long docsSize = docs.getDocsCount();
        for (int d = 0; d < ctxIndexs.length; d++)
        {
        	int[] termIndexs = ctxIndexs[d];
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
            loglikelihood += Math.log(probYX.get(ctxLabelIndexs[d]));
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
            	if (termIndexs[j] >= 0)
            	{
	                int index = modelIndexs[termIndexs[j]][i];
	                if (index >= 0)
	                	lambdaSum += lambda.get(index);
            	}
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
            log.debug("GIS iteration " + epuchNum + " error:" + mean.get(maxIndex) + " likelihood:" + loglikelihood);
			if (progressReporter != null)
			{
				Map<String, Object> params = new HashMap<String, Object>();
				params.put(REPORT_EPOCH_NUM, epuchNum);
				params.put(REPORT_LIKELIDHOOD, loglikelihood);
				params.put(REPORT_ERROR, mean.get(maxIndex));
				progressReporter.progress(params);
			}
            return false;
        }
        log.info("GIS converge on " + mean.get(maxIndex) + " likelihood:" + loglikelihood);
        return true;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void loadModel(MaxentModel model)
    {
        GISModel gisModel = (GISModel) model;
        modelIndexs = gisModel.getModelIndexs();
        labels = gisModel.getLabels();
        termIndexMap = gisModel.getTermIndexMap();
        lambda = gisModel.getLambda();
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
