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

import org.jblas.DoubleMatrix;

import darks.learning.neuron.gradient.GradientComputer;

/**
 * Abstract neuronm network
 * 
 * @author Darks.Liu
 *
 */
public abstract class AbstractNeuronNetwork implements ReConstructon
{

	protected GradientComputer gradComputer;
	
	protected DoubleMatrix weights;
	
	protected DoubleMatrix vBias;
	
	protected DoubleMatrix hBias;
	
	protected DoubleMatrix vInput;
	
	protected DoubleMatrix sigma;
	
	protected DoubleMatrix hiddenSigma;
	
	private NNConfig cfg;

	protected double eps = 1.0e-10;
	
	protected double tolerance = 1.0e-5;
	
	
	
	public void initialize(NNConfig cfg)
	{
		this.cfg = cfg;
	}
	
	public double getLossValue()
	{
		cfg.lossFunction.setInput(vInput);
		cfg.lossFunction.setReConstructon(this);
		return cfg.lossFunction.getLossValue();
	}

	/**
	 * Get gradient through current input values
	 * 
	 * @return GradientComputer
	 */
	public GradientComputer getGradient()
	{
		return getGradient(vInput);
	}

	/**
	 * Get gradient through specify input values
	 * 
	 * @param input Specify input values
	 * @return GradientComputer
	 */
	public abstract GradientComputer getGradient(DoubleMatrix input);

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix reconstruct()
	{
		return reconstruct(vInput);
	}
	

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix reconstruct(DoubleMatrix input)
	{
		return null;
	}
}
