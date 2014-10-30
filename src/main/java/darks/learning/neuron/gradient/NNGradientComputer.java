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
		wOriginGradient = wGrad.dup();
		if (vBiasGrad != null)
		    vOriginGradient = vBiasGrad.dup();
        if (hBiasGrad != null)
            hOriginGradient = hBiasGrad.dup();
		DoubleMatrix wAdaGradLR = null;
		if (config.useAdaGrad)
		{
			buildAdaGrad(wGrad, vBiasGrad, hBiasGrad);
			wAdaGradLR = wAdaGrad.getLearnRates(wGrad);
			wGrad.muli(wAdaGradLR);
			if (hBiasGrad != null && hBiasAdaGrad != null)
			    hBiasGrad = hBiasGrad.mul(hBiasAdaGrad.getLearnRates(hBiasGrad));
            if (vBiasGrad != null && vBiasAdaGrad != null)
                vBiasGrad = vBiasGrad.mul(vBiasAdaGrad.getLearnRates(vBiasGrad));
		}
		else
		{
			wGrad.muli(learnRate);
            if (hBiasGrad != null)
                hBiasGrad = hBiasGrad.mul(learnRate);
            if (vBiasGrad != null)
                vBiasGrad = vBiasGrad.mul(learnRate);
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
        	if (hBiasGrad != null)
        	    hBiasGrad.divi(batchSize);
            if (vBiasGrad != null)
                vBiasGrad.divi(batchSize);
        }
        
        if (config.useRegularization && config.L2 > 0)
        {
        	if (config.useAdaGrad)
        	{
        		wGrad.subi(weights.mul(config.L2).mul(wAdaGradLR));
        	}
        	else
        	{
        		wGrad.subi(weights.mul(config.L2 * learnRate));
        	}
        }

        wGradient = wGrad;
        vGradient = vBiasGrad;
        hGradient = hBiasGrad;
	}
	
	
}
