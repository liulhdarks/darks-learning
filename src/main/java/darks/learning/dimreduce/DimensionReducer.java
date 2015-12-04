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

import java.util.Set;

import darks.learning.corpus.Documents;

public interface DimensionReducer
{
	
	/**
	 * Dimension reduce
	 * @param docs Corpus documents
	 * @param dimension Target dimension
	 * @return Dimension terms
	 */
	public Set<String> dimensionReduction(Documents docs, int dimension);
	
	/**
	 * Dimension reduce by threshold
	 * @param docs Corpus documents
	 * @param threshold Target threshold
	 * @return Dimension terms
	 */
	public Set<String> dimensionReduction(Documents docs, double threshold);
}
