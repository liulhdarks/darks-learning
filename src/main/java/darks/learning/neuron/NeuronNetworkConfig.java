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

import darks.learning.common.rand.JavaRandomFunction;
import darks.learning.common.rand.RandomFunction;
import darks.learning.neuron.activate.ActivateFunction;
import darks.learning.neuron.activate.Activations;

/**
 * Neuron network basic configuration
 * 
 * @author Darks.Liu
 *
 */
public class NeuronNetworkConfig
{

	
	public RandomFunction randomFunction = new JavaRandomFunction();
	
	public ActivateFunction activateFunction = Activations.sigmoid();
	
	public LossFunction lossFunction = LossFunction.lossFunc(LossFunction.NEG_LOGLIKELIHOOD_LOSS, this);
	
	public boolean normalized = false;


	public NeuronNetworkConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public NeuronNetworkConfig setActivateFunction(ActivateFunction activateFunction)
	{
		this.activateFunction = activateFunction;
		return this;
	}

	public NeuronNetworkConfig setLossFunction(LossFunction lossFunction)
	{
		this.lossFunction = lossFunction;
		return this;
	}

	public NeuronNetworkConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}
	
	
}
