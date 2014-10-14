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
package darks.learning.neuron.gradient;

import org.jblas.DoubleMatrix;

import darks.learning.neuron.NNConfig;

/**
 * Gradient updater
 * 
 * @author Darks.Liu
 *
 */
public abstract class GradientComputer
{
	
	int batchSize;
	
	DoubleMatrix wGradient;
	
	DoubleMatrix vGradient;
	
	DoubleMatrix hGradient;
	
	NNConfig config;
	
	public GradientComputer(NNConfig config)
	{
		this.config = config;
	}
	
	/**
	 * Compute gradient values
	 * 
	 * @param wGrad Weights gradient
	 * @param vGrad Visible bias gradient
	 * @param hGrad Hidden bias gradient
	 */
	public abstract void computeGradient(DoubleMatrix wGrad, DoubleMatrix vGrad, DoubleMatrix hGrad);


	public DoubleMatrix getwGradient()
	{
		return wGradient;
	}

	public DoubleMatrix getvGradient()
	{
		return vGradient;
	}

	public DoubleMatrix gethGradient()
	{
		return hGradient;
	}

	public void setBatchSize(int batchSize)
	{
		this.batchSize = batchSize;
	}
}
