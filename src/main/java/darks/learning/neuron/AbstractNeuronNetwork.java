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

import static darks.learning.common.utils.MatrixHelper.log;
import static darks.learning.common.utils.MatrixHelper.sigmoid;
import static darks.learning.common.utils.MatrixHelper.oneMinus;

import org.jblas.DoubleMatrix;

import darks.learning.neuron.gradient.GradientUpdater;

/**
 * Abstract neuronm network
 * 
 * @author Darks.Liu
 *
 */
public abstract class AbstractNeuronNetwork
{

	protected GradientUpdater gradUpdater;
	
	protected DoubleMatrix weights;
	
	protected DoubleMatrix vBias;
	
	protected DoubleMatrix hBias;
	
	protected DoubleMatrix vInput;
	
	private NNConfig cfg;
	
	public void initialize(NNConfig cfg)
	{
		this.cfg = cfg;
	}
	
	public double getLossValue()
	{
		DoubleMatrix preHProb = vInput.mmul(weights).addRowVector(hBias);
		DoubleMatrix sigHidden = sigmoid(preHProb);
		
		DoubleMatrix preVProb = sigHidden.mmul(weights.transpose()).addRowVector(vBias);
		DoubleMatrix sigVisible = sigmoid(preVProb);
		
		double likelihood = vInput.mul(log(sigVisible)).add(
                oneMinus(vInput).mul(log(oneMinus(sigVisible))))
                .rowSums().mean();
		if (cfg.normalized)
		{
			likelihood /= vInput.rows;
		}
		return likelihood;
	}
}
