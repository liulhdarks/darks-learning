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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.neuron.NNConfig;

/**
 * Gradient updater
 * 
 * @author Darks.Liu
 *
 */
public abstract class GradientComputer
{
	
	private static Logger log = LoggerFactory.getLogger(GradientComputer.class);
	
	int batchSize;
	
	DoubleMatrix wOriginGradient;
	
	DoubleMatrix vOriginGradient;
	
	DoubleMatrix hOriginGradient;
	
	DoubleMatrix wGradient;
	
	DoubleMatrix vGradient;
	
	DoubleMatrix hGradient;
	
	DoubleMatrix weights;
	
	DoubleMatrix vBias;
	
	DoubleMatrix hBias;
	
	NNConfig config;
	
	AdaptiveLRGradient wAdaGrad;
	
	AdaptiveLRGradient vBiasAdaGrad;
	
	AdaptiveLRGradient hBiasAdaGrad;
	
	double numIterate;
	
	double learnRate;
	
	public GradientComputer(NNConfig config)
	{
		this.config = config;
		learnRate = config.learnRate;
	}
	
	/**
	 * Compute gradient values
	 */
	public void computeGradient()
	{
		computeGradient(wGradient, vGradient, hGradient);
	}
	
	/**
	 * Compute gradient values
	 */
	public void computeOriginGradient()
	{
		computeGradient(wOriginGradient, vOriginGradient, hOriginGradient);
	}
	
	/**
	 * Compute gradient values
	 * 
	 * @param wGrad Weights gradient
	 * @param vGrad Visible bias gradient
	 * @param hGrad Hidden bias gradient
	 */
	public abstract void computeGradient(DoubleMatrix wGrad, DoubleMatrix vBiasGrad, DoubleMatrix hBiasGrad);

	
	protected void buildAdaGrad(DoubleMatrix wGrad, DoubleMatrix vGrad, DoubleMatrix hGrad)
	{
		if (numIterate == 1)
		{
			if (wGrad != null && wAdaGrad == null)
			{
				wAdaGrad = new AdaptiveLRGradient(wGrad.rows, wGrad.columns);
			}
			if (vGrad != null && vBiasAdaGrad == null)
			{
				vBiasAdaGrad = new AdaptiveLRGradient(vGrad.rows, vGrad.columns);
			}
			if (hGrad != null && hBiasAdaGrad == null)
			{
				hBiasAdaGrad = new AdaptiveLRGradient(hGrad.rows, hGrad.columns);
			}
			if (log.isDebugEnabled())
			{
				log.debug("Build adaptive LR gradient.");
			}
		}
	}
	
	public void override()
	{
		wGradient = wOriginGradient.dup();
		vGradient = vOriginGradient.dup();
		hGradient = hOriginGradient.dup();
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

	public double getNumIterate()
	{
		return numIterate;
	}

	public void setNumIterate(double numIterate)
	{
		this.numIterate = numIterate;
	}

	public double getLearnRate()
	{
		return learnRate;
	}

	public void setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
	}

	public DoubleMatrix getwOriginGradient()
	{
		return wOriginGradient;
	}

	public DoubleMatrix getvOriginGradient()
	{
		return vOriginGradient;
	}

	public DoubleMatrix gethOriginGradient()
	{
		return hOriginGradient;
	}

	public void setwGradient(DoubleMatrix wGradient)
	{
		this.wGradient = wGradient;
	}

	public void setvGradient(DoubleMatrix vGradient)
	{
		this.vGradient = vGradient;
	}

	public void sethGradient(DoubleMatrix hGradient)
	{
		this.hGradient = hGradient;
	}

	public DoubleMatrix getWeights()
	{
		return weights;
	}

	public void setWeights(DoubleMatrix weights)
	{
		this.weights = weights;
	}

	public DoubleMatrix getvBias()
	{
		return vBias;
	}

	public void setvBias(DoubleMatrix vBias)
	{
		this.vBias = vBias;
	}

	public DoubleMatrix gethBias()
	{
		return hBias;
	}

	public void sethBias(DoubleMatrix hBias)
	{
		this.hBias = hBias;
	}
	
}
