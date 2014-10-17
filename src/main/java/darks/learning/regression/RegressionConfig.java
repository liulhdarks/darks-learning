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
package darks.learning.regression;

import darks.learning.LearningConfig;
import darks.learning.common.rand.RandomFunction;
import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.activate.ActivateFunction;
import darks.learning.neuron.activate.Activations;


/**
 * Regression model configuration
 * 
 * @author Darks.Liu
 *
 */
public class RegressionConfig extends LearningConfig
{

	double learnRate;
	
	int maxIteratorCount = 500;
	
	double minError = 0.00001;
	
	boolean randomGradient = false;
	
	ActivateFunction activateFunction = Activations.sigmoid();
	
	boolean useAdaGrad = false;
	
	public RegressionConfig()
	{
		lossType = LossFunction.LOGLIKELIHOOD_LOSS;
		lossFunction = LossFunction.lossFunc(lossType, this);
	}

	/**
	 * Set learn rate
	 * @param learnRate Learn rate
	 * @return this
	 */
	public RegressionConfig setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
		return this;
	}

	public RegressionConfig setMaxIteratorCount(int maxIteratorCount)
	{
		this.maxIteratorCount = maxIteratorCount;
		return this;
	}

	public RegressionConfig setMinError(double minError)
	{
		this.minError = minError;
		return this;
	}

	public RegressionConfig setRandomGradient(boolean randomGradient)
	{
		this.randomGradient = randomGradient;
		return this;
	}
	

	public RegressionConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public RegressionConfig setActivateFunction(ActivateFunction activateFunction)
	{
		this.activateFunction = activateFunction;
		return this;
	}

	public RegressionConfig setLossFunction(LossFunction lossFunction)
	{
		this.lossFunction = lossFunction;
		return this;
	}

	public RegressionConfig setLossType(int lossType)
	{
		this.lossType = lossType;
		lossFunction = LossFunction.lossFunc(lossType, this);
		return this;
	}

	public RegressionConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}

	public RegressionConfig setUseAdaGrad(boolean useAdaGrad)
	{
		this.useAdaGrad = useAdaGrad;
		return this;
	}
	
	
}
