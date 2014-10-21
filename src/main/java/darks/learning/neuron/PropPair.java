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
package darks.learning.neuron;

import org.jblas.DoubleMatrix;

public class PropPair
{
	public DoubleMatrix prob;
	
	public DoubleMatrix sample;

	public PropPair(DoubleMatrix prob, DoubleMatrix sample)
	{
		this.prob = prob;
		this.sample = sample;
	}

	public DoubleMatrix getProb()
	{
		return prob;
	}

	public void setProb(DoubleMatrix prob)
	{
		this.prob = prob;
	}

	public DoubleMatrix getSample()
	{
		return sample;
	}

	public void setSample(DoubleMatrix sample)
	{
		this.sample = sample;
	}
	
	
}
