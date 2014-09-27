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

	public DoubleMatrix feature;

	public DoubleMatrix i2hWeightNeg;
	
	public String name;
	
	public WordNode(String name, DoubleMatrix feature)
	{
		this.name = name;
		this.feature = feature;
	}
    
    public WordNode(String name, int freq, Word2VecConfig cfg)
    {
        this(name, freq, cfg.featureSize, cfg.randomFunction);
    }
	
	public WordNode(String name, int freq, int featureSize, RandomFunction func)
	{
		this.name = name;
		this.value = freq;
		feature = new DoubleMatrix(featureSize);
		i2hWeightNeg = new DoubleMatrix(featureSize);
		for (int i = 0; i < featureSize; i++)
		{
			feature.put(i, (func.randDouble() - 0.5) / featureSize);
			i2hWeightNeg.put(i, (func.randDouble() - 0.5) / featureSize);
		}
	}

    @Override
    public String toString()
    {
        return name;
    }
	
	
}
