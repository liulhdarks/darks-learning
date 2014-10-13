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
public class GradientUpdater
{

	int batchSize;
	
	DoubleMatrix wGradient;
	
	DoubleMatrix vGradient;
	
	DoubleMatrix hGradient;
	
	NNConfig config;
	
	public GradientUpdater(NNConfig config)
	{
		this.config = config;
	}
	
	public void updateGradient(DoubleMatrix wGrad, DoubleMatrix vGrad, DoubleMatrix hGrad)
	{
		wGrad.muli(config.learnRate);
		hGrad.muli(config.learnRate);
        vGrad.muli(config.learnRate);
        
        double momentum = config.momentum;
        if (momentum > 0 && wGradient != null)
        {
        	wGrad.addi(wGradient.mul(momentum).add(wGrad.mul(1 - momentum)));
        }
        if (momentum > 0 && vGradient != null)
        {
        	vGrad.addi(vGradient.mul(momentum).add(vGrad.mul(1 - momentum)));
        }
        if (momentum > 0 && hGradient != null)
        {
        	hGrad.addi(hGradient.mul(momentum).add(hGrad.mul(1 - momentum)));
        }
        
        if (config.normalized)
        {
        	wGrad.divi(batchSize);
        	hGrad.divi(batchSize);
        	vGrad.divi(batchSize);
        }
        
        wGradient = wGrad;
        vGradient = vGrad;
        hGradient = hGrad;
	}

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
