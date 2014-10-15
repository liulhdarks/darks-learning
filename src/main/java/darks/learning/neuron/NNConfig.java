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

import darks.learning.LearningConfig;
import darks.learning.common.rand.RandomFunction;
import darks.learning.optimize.LearningOptimizer.OptimizeType;

public class NNConfig extends LearningConfig
{

	public double learnRate = 0.001;
	
	public double momentum = 0;

	public Integer maxIterateCount = 0;
	
	public Integer maxIterateTime = 0;
	
	public boolean useAdaGrad = true;
	
	public OptimizeType optimizeType = OptimizeType.NONE;
	
	public NNConfig setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
		return this;
	}

	public NNConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public NNConfig setMaxIterateCount(Integer maxIterateCount)
	{
		this.maxIterateCount = maxIterateCount;
		return this;
	}

	public NNConfig setMaxIterateTime(Integer maxIterateTime)
	{
		this.maxIterateTime = maxIterateTime;
		return this;
	}

	public NNConfig setMomentum(double momentum)
	{
		this.momentum = momentum;
		return this;
	}

	public NNConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}

	public NNConfig setUseAdaGrad(boolean useAdaGrad)
	{
		this.useAdaGrad = useAdaGrad;
		return this;
	}

	public NNConfig setOptimizeType(OptimizeType optimizeType)
	{
		this.optimizeType = optimizeType;
		return this;
	}
	
	
}
