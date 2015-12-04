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
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.FreqCount;
import darks.learning.corpus.Documents;
import darks.learning.dimreduce.eval.DimReduceEvalFunction;
import darks.learning.dimreduce.eval.MaxDimReduceEvalFunction;

public abstract class DocumentDimensionReducer implements DimensionReducer
{
	
	private static final Logger log = LoggerFactory.getLogger(DocumentDimensionReducer.class);
	
	protected int threadCount = 0;
	
	protected TreeSet<DimReducerSortBean> termsResultSet = new TreeSet<DimReducerSortBean>();
	
	private DimReduceEvalFunction evalFunction = new MaxDimReduceEvalFunction();
	
	public DocumentDimensionReducer()
	{
		this(1);
	}

	public DocumentDimensionReducer(int threadCount)
	{
		this.threadCount = threadCount;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Set<String> dimensionReduction(Documents docs, int dimension)
	{
		prepare(docs);
		compute();
		int count = 0;
		Set<String> result = new LinkedHashSet<String>();
		for (DimReducerSortBean bean : termsResultSet)
		{
			result.add(bean.term);
			if (++count == dimension)
				break;
		}
		return result;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Set<String> dimensionReduction(Documents docs, double threshold)
	{
		prepare(docs);
		compute();
		Set<String> result = new LinkedHashSet<String>();
		for (DimReducerSortBean bean : termsResultSet)
		{
			if (bean.value >= threshold)
				result.add(bean.term);
			else
				break;
		}
		return result;
	}

	public abstract boolean prepare(Documents docs);
	
	public abstract Collection<String> getTerms();
	
	public abstract Collection<String> getLabels();
	
	public abstract double computeTermLabel(String term, String label);
	
	public void compute()
	{
		if (threadCount <= 1)
			computeSingleThread(getTerms());
		else
			computeMultiThread(getTerms());
	}
	
	private void computeSingleThread(Collection<String> terms)
	{
		int termsCount = terms.size();
		long st = System.currentTimeMillis();
		for (String term : terms)
		{
			computeTerm(term);
			if (System.currentTimeMillis() - st > 10000)
			{
				st = System.currentTimeMillis();
				log.debug("Dimension reduce progress " + termsResultSet.size() + "/" + termsCount);
			}
		}
	}
	
	private void computeMultiThread(Collection<String> terms)
	{
		final AtomicLong count = new AtomicLong(0);
		ExecutorService threadPool = Executors.newFixedThreadPool(threadCount);
		final int termsCount = terms.size();
		final CountDownLatch latch = new CountDownLatch(termsCount);
		for (final String term : terms)
		{
			threadPool.execute(new Runnable()
			{
				@Override
				public void run()
				{
					try
					{
						computeTerm(term);
						long v = count.incrementAndGet();
						if (v % 500 == 0)
							log.debug("Dimension reduce progress " + v + "/" + termsCount);
					}
					finally
					{
						latch.countDown();
					}
				}
			});
		}
		try
		{
			latch.await();
			threadPool.shutdown();
		}
		catch (InterruptedException e)
		{
			e.printStackTrace();
		}
	}
	
	public void computeTerm(String term)
	{
		double value = evalFunction.eval(this, term);
		addTermBean(new DimReducerSortBean(term, value));
	}
	
	private synchronized void addTermBean(DimReducerSortBean bean)
	{
		termsResultSet.add(bean);
	}
	
	protected FreqCount<String> getFreqCount(Map<String, FreqCount<String>> map, String key)
	{
		FreqCount<String> result = map.get(key);
		if (result == null)
		{
			result = new FreqCount<String>();
			map.put(key, result);
		}
		return result;
	}

	public DimReduceEvalFunction getEvalFunction()
	{
		return evalFunction;
	}

	public void setEvalFunction(DimReduceEvalFunction evalFunction)
	{
		this.evalFunction = evalFunction;
	}
	
}
