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

import org.jblas.DoubleMatrix;

import darks.learning.neuron.AbstractNeuronNetwork;
import darks.learning.neuron.gradient.GradientComputer;

/**
 * Hidden layer of multiple layers perceptron.
 * 
 * @author Darks.Liu
 *
 */
public class OutputLayer extends AbstractNeuronNetwork
{

	@Override
	public GradientComputer getGradient(DoubleMatrix input)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DoubleMatrix propForward(DoubleMatrix v)
	{
		// TODO Auto-generated method stub
		return super.propForward(v);
	}

	@Override
	public DoubleMatrix propBackward(DoubleMatrix h)
	{
		// TODO Auto-generated method stub
		return super.propBackward(h);
	}

}
