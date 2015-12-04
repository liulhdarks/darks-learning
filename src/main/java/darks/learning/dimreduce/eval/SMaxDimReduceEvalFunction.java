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
package darks.learning.dimreduce.eval;

import java.util.Collection;
import java.util.Comparator;
import java.util.TreeSet;

import darks.learning.dimreduce.DocumentDimensionReducer;

public class SMaxDimReduceEvalFunction implements DimReduceEvalFunction
{
	
	SortComparator comparator;
	
	public SMaxDimReduceEvalFunction()
	{
		comparator = new SortComparator();
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double eval(DocumentDimensionReducer reducer, String term)
	{
		Collection<String> labels = reducer.getLabels();
		TreeSet<Double> sortSet = new TreeSet<Double>(comparator);
		for (String label : labels)
		{
			double value = reducer.computeTermLabel(term, label);
			sortSet.add(value);
		}
		if (sortSet.isEmpty())
			return 0.;
		else if (sortSet.size() == 1)
			return sortSet.first();
		else
		{
			double max = sortSet.pollFirst();
			double smax = sortSet.pollFirst();
			return max - smax;
		}
	}

	class SortComparator implements Comparator<Double>
	{

		@Override
		public int compare(Double o1, Double o2)
		{
			return Double.compare(o2, o1);
		}
		
	}
}
