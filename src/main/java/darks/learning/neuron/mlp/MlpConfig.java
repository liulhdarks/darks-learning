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
package darks.learning.neuron.mlp;

import darks.learning.common.rand.RandomFunction;
import darks.learning.neuron.NNConfig;
import darks.learning.optimize.LearningOptimizer.OptimizeType;

/**
 * Multiple layers neuron network's configuration
 * 
 * @author Darks.Liu
 *
 */
public class MlpConfig extends NNConfig
{

	int inputLayerSize;
	
	int[] hiddenLayouts;
	
	int outputLayerSize;
	
	public MlpConfig()
	{
		
	}
	
	

	public MlpConfig setInputLayerSize(int inputLayerSize)
	{
		this.inputLayerSize = inputLayerSize;
		return this;
	}



	public MlpConfig setHiddenLayouts(int[] hiddenLayouts)
	{
		this.hiddenLayouts = hiddenLayouts;
		return this;
	}

	public MlpConfig setOutputLayerSize(int outputLayerSize)
	{
		this.outputLayerSize = outputLayerSize;
		return this;
	}
	

	
	public MlpConfig setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
		return this;
	}

	public MlpConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public MlpConfig setMaxIterateCount(Integer maxIterateCount)
	{
		this.maxIterateCount = maxIterateCount;
		return this;
	}

	public MlpConfig setMaxIterateTime(Integer maxIterateTime)
	{
		this.maxIterateTime = maxIterateTime;
		return this;
	}

	public MlpConfig setMomentum(double momentum)
	{
		this.momentum = momentum;
		return this;
	}

	public MlpConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}

	public MlpConfig setUseAdaGrad(boolean useAdaGrad)
	{
		this.useAdaGrad = useAdaGrad;
		return this;
	}

	public MlpConfig setOptimizeType(OptimizeType optimizeType)
	{
		this.optimizeType = optimizeType;
		return this;
	}
	
    public MlpConfig setUseRegularization(boolean useRegularization)
    {
        this.useRegularization = useRegularization;
        return this;
    }

    public MlpConfig setL2(double l2)
    {
        L2 = l2;
        return this;
    }
    
}
