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
package darks.learning.neuron.da;

import darks.learning.common.rand.RandomFunction;
import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.NNConfig;
import darks.learning.neuron.activate.ActivateFunction;
import darks.learning.neuron.activate.Activations;
import darks.learning.optimize.LearningOptimizer.OptimizeType;

/**
 * Denoising encoder configuration
 * 
 * @author Darks.Liu
 *
 */
public class AutoEncoderConfig extends NNConfig
{
	
	boolean concatBias = false;
	
	int visibleSize;
	
	int hiddenSize;
	
	double corruptionLevel = 0.3;
	
	ActivateFunction activateFunction = Activations.sigmoid();
	
	public AutoEncoderConfig()
	{
		setLossType(LossFunction.RECONSTRUCTION_CROSSENTROPY);
	}

	public AutoEncoderConfig setVisibleSize(int visibleSize)
	{
		this.visibleSize = visibleSize;
		return this;
	}

	public AutoEncoderConfig setHiddenSize(int hiddenSize)
	{
		this.hiddenSize = hiddenSize;
		return this;
	}

	public AutoEncoderConfig setLossType(int lossType)
	{
		this.lossType = lossType;
		this.lossFunction = LossFunction.lossFunc(lossType, this);
		return this;
	}

	public AutoEncoderConfig setConcatBias(boolean concatBias)
	{
		this.concatBias = concatBias;
		return this;
	}
	
	public AutoEncoderConfig setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
		return this;
	}

	public AutoEncoderConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public AutoEncoderConfig setMaxIterateCount(Integer maxIterateCount)
	{
		this.maxIterateCount = maxIterateCount;
		return this;
	}

	public AutoEncoderConfig setMaxIterateTime(Integer maxIterateTime)
	{
		this.maxIterateTime = maxIterateTime;
		return this;
	}

	public AutoEncoderConfig setMomentum(double momentum)
	{
		this.momentum = momentum;
		return this;
	}

	public AutoEncoderConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}

	public AutoEncoderConfig setUseAdaGrad(boolean useAdaGrad)
	{
		this.useAdaGrad = useAdaGrad;
		return this;
	}

	public AutoEncoderConfig setOptimizeType(OptimizeType optimizeType)
	{
		this.optimizeType = optimizeType;
		return this;
	}

	public AutoEncoderConfig setCorruptionLevel(double corruptionLevel)
	{
		this.corruptionLevel = corruptionLevel;
		return this;
	}

	public AutoEncoderConfig setActivateFunction(ActivateFunction activateFunction)
	{
		this.activateFunction = activateFunction;
		return this;
	}
	
	
}
