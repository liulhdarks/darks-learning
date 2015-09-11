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
package darks.learning.classifier.bayes;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.FreqCount;
import darks.learning.corpus.Documents;
import darks.learning.corpus.Documents.Document;

/**
 * Native Bayes algorithm
 * 
 * @author Darks.Liu
 *
 */
public class NaiveBayes
{
	
	private static Logger log = LoggerFactory.getLogger(NaiveBayes.class);
	
	public static final int BINAMIAL = 0;
	
	public static final int BERNOULLI = 1;
	
	public static final int LAPLACE = 1;
	
	public NaiveBayesConfig config = new NaiveBayesConfig();
	
	long uniqueTermCount = 0;
	
	long totalDocCount = 0;
	
	Map<String, List<Document>> labelsMap;
	
	Map<String, FreqCount<String>> labelsFreqMap;
	
	Map<String, Map<String, Set<Document>>> labelsTermDocMap;
	
	FreqCount<String> termsFreq;
	
	public NaiveBayes()
	{
		
	}

	/**
	 * Train naive bayes
	 * @param docs Documents
	 */
	public void train(Documents docs)
	{
		labelsMap = docs.getLabelsMap();
		termsFreq = docs.getTermsFreq();
		uniqueTermCount = termsFreq.getUniqueCount();
		labelsFreqMap = new HashMap<String, FreqCount<String>>();
		if (config.modelType == BERNOULLI)
		{
			labelsTermDocMap = new HashMap<String, Map<String, Set<Document>>>();
		}
		log.info("Training native bayes model.Labels:" + labelsMap.size() + " terms:" + termsFreq.getUniqueCount());
		for (Entry<String, List<Document>> entry : labelsMap.entrySet())
		{
			String label = entry.getKey();
			FreqCount<String> freq = labelsFreqMap.get(label);
			if (freq == null)
			{
				freq = new FreqCount<String>();
				labelsFreqMap.put(label, freq);
			}
			for (Document doc : entry.getValue())
			{
				for (String term : doc.getTerms())
				{
					freq.addValue(term);
					addTermDocument(label, term, doc);
				}
				totalDocCount++;
			}
		}
		log.info("Complete to train native bayes model.");
	}
	
	private void addTermDocument(String label, String term, Document doc)
	{
		if (config.modelType != BERNOULLI)
		{
			return;
		}
		Map<String, Set<Document>> termMap = labelsTermDocMap.get(label);
		if (termMap == null)
		{
			termMap = new HashMap<String, Set<Document>>();
			labelsTermDocMap.put(label, termMap);
		}
		Set<Document> docs = termMap.get(term);
		if (docs == null)
		{
			docs = new HashSet<Document>();
			termMap.put(term, docs);
		}
		docs.add(doc);
	}
	
	private int getTermDocCount(String label, String term)
	{
		Map<String, Set<Document>> termMap = labelsTermDocMap.get(label);
		if (termMap == null)
		{
			return 0;
		}
		Set<Document> docs = termMap.get(term);
		if (docs == null)
		{
			return 0;
		}
		return docs.size();
	}
	
	public String predict(String sentence)
	{
		return predict(Arrays.asList(sentence.trim().split(" ")));
	}
	
	public String predict(String[] terms)
	{
		return predict(Arrays.asList(terms));
	}
	
	public String predict(List<String> terms)
	{
		if (terms.isEmpty())
		{
			return null;
		}
		double maxProb = 0;
		String maxLalel = null;
		for (String label : labelsMap.keySet())
		{
			double prob = 0;
			switch (config.modelType)
			{
			case BINAMIAL:
				prob = binomial(label, terms);
				break;

			case BERNOULLI:
				prob = bernoulli(label, terms);
				break;
			}
			if (prob > maxProb || maxLalel == null)
			{
				maxProb = prob;
				maxLalel = label;
			}
		}
		return maxLalel;
	}
	
	private double binomial(String label, List<String> terms)
	{
		FreqCount<String> labelTermTotalCount = labelsFreqMap.get(label);
		double pC = (double)labelTermTotalCount.totalCount() / (double)termsFreq.totalCount();
		Double pTC = null;
		for (String term : terms)
		{
			Double pTCi = binomialProbality(label, term);
			if (pTCi == null)
			{
				continue;
			}
			if (config.logLikelihood)
			{
				pTC = pTC == null ? Math.log(pTCi) : pTC + Math.log(pTCi);
			}
			else
			{
				pTC = pTC == null ? pTCi : pTC * pTCi;
			}
		}
		if (pTC == null)
		{
			pTC = 0.0001;
		}
		return config.logLikelihood ? pTC + Math.log(pC) : pTC * pC;
	}
	
	
	private Double binomialProbality(String label, String term)
	{
		FreqCount<String> freq = labelsFreqMap.get(label);
		if (freq == null)
		{
			return null;
		}
		Long termLabelCount = freq.getValue(term);
		return (double)(termLabelCount + 1) / (double)(freq.totalCount() + uniqueTermCount);
	}
	
	private double bernoulli(String label, List<String> terms)
	{
		double pC = (double)labelsMap.get(label).size() / (double)totalDocCount;
		Double pTC = null;
		Set<String> berTerms = new HashSet<String>(terms);
		Iterator<Entry<String, Long>> it = termsFreq.entrySetIterator();
		while (it.hasNext())
		{
			Entry<String, Long> entry = it.next();
			Double pTCi = bernoulliProbality(label, entry.getKey());
			if (pTCi == null)
			{
				continue;
			}
			if (!berTerms.contains(entry.getKey()))
			{
				pTCi = 1 - pTCi;
			}
			if (config.logLikelihood)
			{
				pTC = pTC == null ? Math.log(pTCi) : pTC + Math.log(pTCi);
			}
			else
			{
				pTC = pTC == null ? pTCi : pTC * pTCi;
			}
		}
		if (pTC == null)
		{
			pTC = 0.0001;
		}
		return config.logLikelihood ? pTC + Math.log(pC) : pTC * pC;
	}
	
	
	private Double bernoulliProbality(String label, String term)
	{
		FreqCount<String> freq = labelsFreqMap.get(label);
		if (freq == null)
		{
			return null;
		}
		List<Document> docs = labelsMap.get(label);
		if (docs == null)
		{
			return null;
		}

		int docCount = getTermDocCount(label, term);
		return (double)(docCount + 1) / (double)(docs.size() + 2);
	}
	
}
