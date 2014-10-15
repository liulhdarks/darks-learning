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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.neuron.AbstractNeuronNetwork;
import darks.learning.neuron.gradient.GradientComputer;

/**
 * Iterate training without being optimized
 * 
 * @author Darks.Liu
 *
 */
public class NoneNeuronNetworkOptimizer extends AbstractNeuronNetworkOptimizer
{

	private static Logger log = LoggerFactory.getLogger(NoneNeuronNetworkOptimizer.class);

	protected double eps = 1.0e-10;
	
	protected double tolerance = 1.0e-5;
	
	public NoneNeuronNetworkOptimizer(AbstractNeuronNetwork network)
	{
		super(network);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void optimize()
	{
		startTime = System.currentTimeMillis();
		int numIterate = 1;
		double lastLoss = 0;
		int iterCount = config.maxIterateCount;
		while (iterCount == 0 || numIterate < iterCount)
		{
			iterate(numIterate);
			double loss = network.getLossValue();
			if (numIterate > 1)
			{
				if (2.0 * Math.abs(loss - lastLoss) <= tolerance * (Math.abs(loss) + Math.abs(lastLoss) + eps)) 
				{
	                log.info ("Gradient Ascent: Value difference " + Math.abs(loss - lastLoss) +" below " +
	                        "tolerance; arriving converged.");
	                break;
	            }
			}
			lastLoss = loss;
			if (log.isDebugEnabled())
			{
				log.debug("Iteration number " + numIterate + " loss:" + loss);
			}
			if (!checkIterateTime())
			{
				break;
			}
			numIterate++;
		}
	}
	
	private void iterate(int numIterate)
	{
		network.setNumIterate(numIterate);
		GradientComputer grad = network.getGradient();
		network.addGrad(grad);
	}

}
