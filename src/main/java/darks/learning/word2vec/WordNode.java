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
package darks.learning.word2vec;

import org.jblas.DoubleMatrix;

import darks.learning.common.basic.HaffNode;
import darks.learning.common.rand.RandomFunction;

public class WordNode extends HaffNode
{

	public DoubleMatrix i2hWeight;

	public DoubleMatrix i2hWeightNeg;
	
	public String value;
	
	public int freq;
	
	public WordNode(String value, int freq, Word2VecConfig cfg)
	{
		this.value = value;
		this.freq = freq;
		int featureSize = cfg.featureSize;
		i2hWeight = new DoubleMatrix(featureSize);
		i2hWeightNeg = new DoubleMatrix(featureSize);
		RandomFunction func = cfg.randomFunction;
		for (int i = 0; i < featureSize; i++)
		{
			i2hWeight.put(i, (func.randDouble() - 0.5) / featureSize);
			i2hWeightNeg.put(i, (func.randDouble() - 0.5) / featureSize);
		}
	}
}
