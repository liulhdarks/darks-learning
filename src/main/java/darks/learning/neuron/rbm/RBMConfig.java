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
package darks.learning.neuron.rbm;

import darks.learning.common.rand.RandomFunction;
import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.NNConfig;
import darks.learning.optimize.LearningOptimizer.OptimizeType;


/**
 * Restricted boltzmann machine algorithm configuration
 * 
 * @author Darks.Liu
 *
 */
public class RBMConfig extends NNConfig
{
	
	public class LayoutType
	{
		public static final int BINARY = 0;
		
		public static final int GAUSSION = 1;
		
		public static final int RECTIFIED = 2;
		
		public static final int SOFTMAX = 3;
	}
	
	public int visibleSize;
	
	public int hiddenSize;
	
	public int visibleType = LayoutType.BINARY;
	
	public int hiddenType = LayoutType.BINARY;
	
	public boolean concatBias = false;
	
	public int gibbsCount = 1;
	
	public RBMConfig()
	{
		lossFunction = LossFunction.lossFunc(LossFunction.RECONSTRUCTION_CROSSENTROPY, this);
	}

	public RBMConfig setVisibleSize(int visibleSize)
	{
		this.visibleSize = visibleSize;
		return this;
	}

	public RBMConfig setHiddenSize(int hiddenSize)
	{
		this.hiddenSize = hiddenSize;
		return this;
	}

	public RBMConfig setVisibleType(int visibleType)
	{
		this.visibleType = visibleType;
		return this;
	}

	public RBMConfig setHiddenType(int hiddenType)
	{
		this.hiddenType = hiddenType;
		return this;
	}

	public RBMConfig setLayoutType(int layoutType)
	{
		this.visibleType = layoutType;
		this.hiddenType = layoutType;
		return this;
	}

	public RBMConfig setConcatBias(boolean concatBias)
	{
		this.concatBias = concatBias;
		return this;
	}

	public RBMConfig setGibbsCount(int gibbsCount)
	{
		this.gibbsCount = gibbsCount;
		return this;
	}

	public RBMConfig setLossType(int lossType)
	{
		this.lossFunction = LossFunction.lossFunc(lossType, this);
		return this;
	}
	
	public RBMConfig setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
		return this;
	}

	public RBMConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public RBMConfig setMaxIterateCount(Integer maxIterateCount)
	{
		this.maxIterateCount = maxIterateCount;
		return this;
	}

	public RBMConfig setMaxIterateTime(Integer maxIterateTime)
	{
		this.maxIterateTime = maxIterateTime;
		return this;
	}

	public RBMConfig setMomentum(double momentum)
	{
		this.momentum = momentum;
		return this;
	}

	public RBMConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}

	public RBMConfig setUseAdaGrad(boolean useAdaGrad)
	{
		this.useAdaGrad = useAdaGrad;
		return this;
	}

	public RBMConfig setOptimizeType(OptimizeType optimizeType)
	{
		this.optimizeType = optimizeType;
		return this;
	}
	
	
}
