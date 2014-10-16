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
package darks.learning.optimize.linesearch;

import org.jblas.DoubleMatrix;

import darks.learning.neuron.AbstractNeuronNetwork;

/**
 * Line search algorithm
 * 
 * @author Darks.Liu
 *
 */
public abstract class LineSearch
{

	protected AbstractNeuronNetwork network;
	
	protected int maxIterateCount = 100;
	
	public LineSearch(AbstractNeuronNetwork network)
	{
		this.network = network;
	}
	
	/**
	 * Optimize learning step
	 * 
	 * @param x Input x
	 * @param numIterate Iteration number
	 * @param initStep Initialize step value
	 * @return New step value
	 */
	public abstract double optimize(DoubleMatrix x, int numIterate, double initStep);
}
