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
package darks.learning.dimreduce;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import darks.learning.common.utils.FreqCount;
import darks.learning.corpus.Documents;
import darks.learning.corpus.Documents.Document;

public class CHI extends DocumentDimensionReducer
{
	
	Map<String, FreqCount<String>> termFreq = new HashMap<String, FreqCount<String>>();

	Map<String, FreqCount<String>> labelFreq = new HashMap<String, FreqCount<String>>();
	
	Set<String> labels = new HashSet<String>();
	
	long totalCount = 0;

	
	public CHI()
	{
		super();
	}

	public CHI(int threadCount)
	{
		super(threadCount);
	}
	
	@Override
	public double computeTermLabel(String term, String label)
	{
		long A = 0;
		long B = 0;
		long C = 0;
		long D = 0;
		FreqCount<String> freq = termFreq.get(term);
		Iterator<Entry<String, Long>> it = freq.entrySetIterator();
		while (it.hasNext())
		{
			Entry<String, Long> freqEntry = it.next();
			if (freqEntry.getKey().equals(label))
			{
				A = freqEntry.getValue();
			}
			else
			{
				B += freqEntry.getValue();
			}
		}
		
		freq = labelFreq.get(label);
		it = freq.entrySetIterator();
		while (it.hasNext())
		{
			Entry<String, Long> freqEntry = it.next();
			if (!freqEntry.getKey().equals(term))
			{
				C += freqEntry.getValue();
			}
		}
		
		for (Entry<String, FreqCount<String>> entry : termFreq.entrySet())
		{
			String etTerm = entry.getKey();
			if (etTerm.equals(term))
				continue;
			FreqCount<String> etFreq = entry.getValue();
			Iterator<Entry<String, Long>> etIt = etFreq.entrySetIterator();
			while (etIt.hasNext())
			{
				Entry<String, Long> freqEntry = etIt.next();
				if (!freqEntry.getKey().equals(label))
				{
					D += freqEntry.getValue();
				}
			}
		}
		return compute(A, B, C, D);
	}
	
	private double compute(long A, long B, long C, long D)
	{
		return (double) totalCount * (double) Math.pow(A * D - B * C, 2) 
				/ (double) (A + C) * (B + D) * (A + B) * (C + D);
	}
	
	@Override
	public boolean prepare(Documents docs)
	{
		Map<String, List<Document>> labelsMap = docs.getLabelsMap();
		for (Entry<String, List<Document>> entry : labelsMap.entrySet())
		{
			String label = entry.getKey();
			labels.add(label);
			FreqCount<String> termsCount = getFreqCount(labelFreq, label);
			List<Document> list = entry.getValue();
			for (Document doc : list)
			{
				List<String> terms = doc.getTerms();
				for (String term : terms)
				{
					FreqCount<String> labelCount = getFreqCount(termFreq, term);
					termsCount.addValue(term);
					labelCount.addValue(label);
				}
				totalCount++;
			}
		}
		return true;
	}

	@Override
	public Collection<String> getTerms()
	{
		return termFreq.keySet();
	}

	@Override
	public Collection<String> getLabels()
	{
		return labels;
	}
}
