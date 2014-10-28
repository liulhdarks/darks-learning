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

import darks.learning.LearningConfig;
import darks.learning.common.rand.RandomFunction;
import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.activate.ActivateFunction;
import darks.learning.neuron.activate.Activations;

public class LayerConfig extends LearningConfig
{
	
	int layerSize;
	
	ActivateFunction activateFunction = Activations.sigmoid();
	
	public LayerConfig()
	{
		
	}
	
	
	public LayerConfig setLayerSize(int layerSize)
	{
		this.layerSize = layerSize;
		return this;
	}



	public LayerConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public LayerConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}

	public LayerConfig setLossFunction(LossFunction lossFunction)
	{
		this.lossFunction = lossFunction;
		return this;
	}

	public LayerConfig setLossType(int lossType)
	{
		this.lossType = lossType;
		return this;
	}

	public LayerConfig setUseRegularization(boolean useRegularization)
	{
		this.useRegularization = useRegularization;
		return this;
	}

	public LayerConfig setL2(double l2)
	{
		this.L2 = l2;
		return this;
	}
	
	
}
