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
public class NNGradientComputer extends GradientComputer
{
	
	public NNGradientComputer(NNConfig config)
	{
		super(config);
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public void computeGradient(DoubleMatrix wGrad, DoubleMatrix vBiasGrad, DoubleMatrix hBiasGrad)
	{
		if (config.useAdaGrad)
		{
			buildAdaGrad(wGrad, vBiasGrad, hBiasGrad);
			wGrad.muli(wAdaGrad.getLearnRates(wGrad));
			hBiasGrad = hBiasGrad.mul(hBiasAdaGrad.getLearnRates(hBiasGrad));
			vBiasGrad = vBiasGrad.mul(vBiasAdaGrad.getLearnRates(vBiasGrad));
		}
		else
		{
			wGrad.muli(config.learnRate);
			hBiasGrad = hBiasGrad.mul(config.learnRate);
			vBiasGrad = vBiasGrad.mul(config.learnRate);
		}
        
        double momentum = config.momentum;
        if (momentum > 0 && wGradient != null)
        {
        	wGrad.addi(wGradient.mul(momentum).add(wGrad.mul(1 - momentum)));
        }
        if (momentum > 0 && vGradient != null)
        {
        	vBiasGrad.addi(vGradient.mul(momentum).add(vBiasGrad.mul(1 - momentum)));
        }
        if (momentum > 0 && hGradient != null)
        {
        	hBiasGrad.addi(hGradient.mul(momentum).add(hBiasGrad.mul(1 - momentum)));
        }
        
        if (config.normalized)
        {
        	wGrad.divi(batchSize);
        	hBiasGrad.divi(batchSize);
        	vBiasGrad.divi(batchSize);
        }
        
        wGradient = wGrad;
        vGradient = vBiasGrad;
        hGradient = hBiasGrad;
	}
	
	
}
