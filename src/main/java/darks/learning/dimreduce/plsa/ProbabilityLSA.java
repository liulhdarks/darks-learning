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
package darks.learning.dimreduce.plsa;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.basic.TfIdf;
import darks.learning.common.rand.JdkRandomFunction;
import darks.learning.common.rand.RandomFunction;
import darks.learning.common.utils.FreqCount;
import darks.learning.corpus.Corpus;

/**
 * Probabilistic Latent Semantic Analysis
 * @author Darks.Liu
 *
 */
public class ProbabilityLSA
{

    private static Logger log = LoggerFactory.getLogger(ProbabilityLSA.class);
    
    private final static double MAGICNUM = 0.0000000000000001;

    private int topicSize = 300;
    
    RandomFunction randFunc = new JdkRandomFunction(System.currentTimeMillis());
    
    Map<String, Integer> wordIndexMap = null;
    
    Map<String, Integer> docIndexMap = null;
    
    int iterNumber = 100;
    
    int docSize;
    
    int termSize;
    
    DoubleMatrix proDocTopic;
    
    DoubleMatrix proTopicTerm;
    
    DoubleMatrix[] proDocTermTopics;
    
    int[][] docTermMatrix;
    
    public ProbabilityLSA()
    {
    	
    }
    
    /**
     * Construction
     * @param targetDimension Target reduce dimension.Default 300
     */
    public ProbabilityLSA(int targetDimension)
    {
    	topicSize = targetDimension > 0 ? targetDimension : topicSize;
    }
    
    /**
     * Construction
     * @param targetDimension Target reduce dimension.Default 300
     * @param customRandFunc Custom random function.Default {@linkplain darks.learning.common.rand.JdkRandomFunction JdkRandomFunction}
     */
    public ProbabilityLSA(int targetDimension, RandomFunction customRandFunc)
    {
    	topicSize = targetDimension > 0 ? targetDimension : topicSize;
    	randFunc = customRandFunc != null ? customRandFunc : randFunc;
    }

    /**
     * Train model by corpus
     * @param corpus {@linkplain darks.learning.corpus.Corpus Corpus}
     */
    public void train(Corpus corpus)
    {
    	prepareParam(corpus);
    	for (int i = 1; i <= iterNumber; i++)
    	{
        	eStep();
        	mStep();
    		if (checkConverge(i))
    		{
    			break;
    		}
    	}
    }
    
    private void prepareParam(Corpus corpus)
    {
    	wordIndexMap = new HashMap<String, Integer>();
    	docIndexMap = new HashMap<String, Integer>();
    	TfIdf tfidf = corpus.getTfIDF();
        int docCount = (int)tfidf.getTotalSentenceCount();
        int termsCount = (int)tfidf.getUniqueWordsCount();
        docTermMatrix = new int[docCount][termsCount];
    	int rowIndex = 0;
    	int wordIndexSeed = 0;
        for (Entry<String, FreqCount<String>> entryFreq : tfidf.getSentenceMap().entrySet())
        {
        	String docContent = entryFreq.getKey();
            FreqCount<String> freq = entryFreq.getValue();
            docIndexMap.put(docContent, rowIndex);
            Iterator<Map.Entry<String, Long>> it = freq.entrySetIterator();
            while (it.hasNext())
            {
                Entry<String, Long> entry = it.next();
                String word = (String)entry.getKey();
                Integer wordIndex = wordIndexMap.get(word);
                if (wordIndex == null)
                {
                	wordIndex = wordIndexSeed++;
                	wordIndexMap.put(word, wordIndex);
                }
                docTermMatrix[rowIndex][wordIndex]++;
            }
            rowIndex++;
        }
        docSize = docCount;
        termSize = termsCount;

    	proDocTopic = randMatrix(docSize, topicSize);
    	proTopicTerm = randMatrix(topicSize, termSize);
    	proDocTermTopics = new DoubleMatrix[docSize];
    	for (int i = 0; i < docSize; i++)
    	{
    		proDocTermTopics[i] = new DoubleMatrix(termSize, topicSize);
    	}
    }
    
    private DoubleMatrix randMatrix(int rowSize, int colSize)
    {
    	DoubleMatrix result = DoubleMatrix.rand(rowSize, colSize); 
    	DoubleMatrix rowSum = result.rowSums();
    	result.diviColumnVector(rowSum);
    	return result;
    }
    
    private void eStep()
    {
    	for (int d = 0; d < docSize; d++)
    	{
    		for (int w = 0; w < termSize; w++)
        	{
    			double totalSum = 0;
        		for (int z = 0; z < topicSize; z++)
            	{
        			double proDocTermTopic = proDocTopic.get(d, z) * proTopicTerm.get(z, w);
        			totalSum += proDocTermTopic;
        			proDocTermTopics[d].put(w, z, proDocTermTopic);
            	}
        		totalSum = Double.compare(totalSum, 0.0) == 0 ? MAGICNUM : totalSum;
        		for (int z = 0; z < topicSize; z++)
            	{
        			proDocTermTopics[d].put(w, z, proDocTermTopics[d].get(w, z) / totalSum);
            	}
        	}
    	}
    }
    
    private void mStep()
    {
    	//p(w|z)
		for (int z = 0; z < topicSize; z++)
    	{
			double totalSum = 0;
    		for (int w = 0; w < termSize; w++)
        	{
    			double p = 0;
    	    	for (int d = 0; d < docSize; d++)
    	    	{
	    	    	double v = docTermMatrix[d][w] * proDocTermTopics[d].get(w, z);
	    	    	p += v;
    	    	}
    	    	proTopicTerm.put(z, w, p);
    	    	totalSum += p;
	    	}
    		totalSum = Double.compare(totalSum, 0.0) == 0 ? MAGICNUM : totalSum;
    		for (int w = 0; w < termSize; w++)
        	{
    			double v = proTopicTerm.get(z, w) / totalSum;
    			proTopicTerm.put(z, w, v);
        	}
    	}
		
		//p(z|d)
    	for (int d = 0; d < docSize; d++)
    	{
			double totalSum = 0;
    		for (int z = 0; z < topicSize; z++)
        	{
    			double p = 0;
        		for (int w = 0; w < termSize; w++)
            	{
        			double v = docTermMatrix[d][w] * proDocTermTopics[d].get(w, z);
        			p += v;
            	}
        		proDocTopic.put(d, z, p);
    	    	totalSum += p;
        	}
    		totalSum = Double.compare(totalSum, 0.0) == 0 ? MAGICNUM : totalSum;
    		for (int z = 0; z < topicSize; z++)
        	{
    			double v = proDocTopic.get(d, z) / totalSum;
    			proDocTopic.put(d, z, v);
        	}
    	}
    }
    
    private boolean checkConverge(int iterNumber)
    {
    	double logVal = 0;
    	for (int d = 0; d < docSize; d++)
    	{
    		for (int w = 0; w < termSize; w++)
        	{
    			double sum = 0;
        		for (int z = 0; z < topicSize; z++)
            	{
        			double proDocTermTopic = proDocTopic.get(d, z) * proTopicTerm.get(z, w);
        			sum += proDocTermTopic;
            	}
        		if (Double.compare(sum, 0.0) == 0)
        			sum = MAGICNUM;
        		logVal += docTermMatrix[d][w] * Math.log10(sum);
        	}
    	}
    	log.debug("Iteration:" + iterNumber + " Loglikelihood value:" + logVal);
    	return false;
    }



    /**
     * Predict target sentence content by words' array
     * @param words Words' array
     * @return Target sentence content
     */
    public String predict(String[] words)
    {
        int column = predictIndex(words);
        return null;
    }

    /**
     * Predict target index by words' array
     * @param words Words' array
     * @return Target index
     */
    public int predictIndex(String[] words)
    {
        return 0;
    }
    
    public double distance(Collection<String> sources, Collection<String> targets)
	{
		DoubleMatrix srcMt = getSentenceFeature(sources);
		DoubleMatrix targetMt = getSentenceFeature(targets);
		if (srcMt == null || targetMt == null)
		{
			return 0.001;
		}
		double sim = srcMt.dot(targetMt);
		sim = sim / (srcMt.norm2() * targetMt.norm2());
		return sim;
	}
    
    public double distanceDocuments(String doc1, String doc2)
    {
    	Integer docIndex = docIndexMap.get(doc1);
    	Integer docIndex2 = docIndexMap.get(doc2);
		if (docIndex == null || docIndex2 == null)
		{
			return 0.001;
		}
		DoubleMatrix srcMt = proDocTopic.getRow(docIndex);
		DoubleMatrix targetMt = proDocTopic.getRow(docIndex2);
		double sim = srcMt.dot(targetMt);
		sim = sim / (srcMt.norm2() * targetMt.norm2());
		return sim;
    }
    
    private DoubleMatrix getProTermTopic(String word)
    {
    	Integer wordIndex = wordIndexMap.get(word);
    	if (wordIndex == null)
    		return null;
    	return proTopicTerm.getColumn(wordIndex);
    }
    
    public DoubleMatrix getSentenceFeature(Collection<String> words)
	{
		DoubleMatrix center = null;
		for (String word : words)
		{
			DoubleMatrix feature = getProTermTopic(word);
			if (feature == null)
				continue;
			feature = feature.dup();
			if (center == null)
			{
				center = feature;
			}
			else
			{
				center.addi(feature);
			}
		}
		return center;
	}

	public int getIterNumber()
	{
		return iterNumber;
	}

	public void setIterNumber(int iterNumber)
	{
		this.iterNumber = iterNumber;
	}
    
    
}
