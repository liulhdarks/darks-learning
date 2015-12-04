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
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import darks.learning.common.utils.FreqCount;
import darks.learning.corpus.Documents;
import darks.learning.corpus.Documents.Document;

public class CD extends DocumentDimensionReducer
{
	
	private Map<String, FreqCount<String>> termFreqMap = new HashMap<String, FreqCount<String>>();
	
	private Map<String, FreqCount<String>> docFreqMap = new HashMap<String, FreqCount<String>>();
	
	private Map<String, Long> labelDocMap = new HashMap<String, Long>();

	
	
	public CD()
	{
		super();
	}

	public CD(int threadCount)
	{
		super(threadCount);
	}

	@Override
	public boolean prepare(Documents docs)
	{
		Map<String, List<Document>> labelsMap = docs.getLabelsMap();
		for (Entry<String, List<Document>> entry : labelsMap.entrySet())
		{
			String label = entry.getKey();
			List<Document> list = entry.getValue();
			labelDocMap.put(label, (long)list.size());
			for (Document doc : list)
			{
				Set<String> repeat = new HashSet<String>();
				List<String> terms = doc.getTerms();
				for (String term : terms)
				{
					FreqCount<String> termFreq = getFreqCount(termFreqMap, term);
					termFreq.addValue(label);
					if (!repeat.contains(term))
					{
						FreqCount<String> docFreq = getFreqCount(docFreqMap, term);
						docFreq.addValue(label);
						repeat.add(term);
					}
				}
			}
		}
		return true;
	}
	
	@Override
	public double computeTermLabel(String term, String label)
	{
		FreqCount<String> docFreq = getFreqCount(docFreqMap, term);
		long df = docFreq.getValue(label);
		long totalDf = docFreq.totalCount();

		FreqCount<String> termFreq = getFreqCount(termFreqMap, term);
		long tf = termFreq.getValue(label);
		long totalTf = termFreq.totalCount();
		
		long C = labelDocMap.get(label);
		return ((double) df / (double) totalDf) * ((double) tf / (double) totalTf) * ((double) df / (double) C);
	}

	@Override
	public Collection<String> getTerms()
	{
		return termFreqMap.keySet();
	}

	@Override
	public Collection<String> getLabels()
	{
		return labelDocMap.keySet();
	}
}
