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

import darks.learning.dimreduce.DocumentDimensionReducer;

public class MaxDimReduceEvalFunction implements DimReduceEvalFunction
{

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double eval(DocumentDimensionReducer reducer, String term)
	{
		Collection<String> labels = reducer.getLabels();
		Double max = null;
		for (String label : labels)
		{
			double value = reducer.computeTermLabel(term, label);
			if (max == null || value > max)
				max = value;
		}
		return max == null ? 0. : max; 
	}

}
