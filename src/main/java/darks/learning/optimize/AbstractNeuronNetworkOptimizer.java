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
package darks.learning.optimize;

import darks.learning.neuron.AbstractNeuronNetwork;
import darks.learning.neuron.NNConfig;

/**
 * Iterate training without being optimized
 * 
 * @author Darks.Liu
 *
 */
public abstract class AbstractNeuronNetworkOptimizer implements LearningOptimizer
{

	protected AbstractNeuronNetwork network;
	
	protected NNConfig config;
	
	protected long startTime;
	
	public AbstractNeuronNetworkOptimizer(AbstractNeuronNetwork network)
	{
		this.network = network;
		config = network.config();
	}

	protected boolean checkIterateTime()
	{
		if (config.maxIterateTime > 0 
				&& System.currentTimeMillis() - startTime >= config.maxIterateTime)
		{
			return false;
		}
		return true;
	}

}
