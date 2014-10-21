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
package darks.learning.lossfunc;

import org.jblas.DoubleMatrix;

import darks.learning.LearningConfig;
import darks.learning.neuron.ReConstructon;

/**
 * Loss function
 * 
 * @author Darks.Liu
 *
 */
public abstract class LossFunction
{
	
	/**
	 * Squared loss function
	 */
	public static final int SQUARED_LOSS = 1;
	
	/**
	 * Negative log likelihood loss function.
	 * y*log(h(x))+(1-y)*log(1-h(x))
	 */
	public static final int LOGLIKELIHOOD_LOSS = 2;
	
	public static final int RECONSTRUCTION_CROSSENTROPY = 3;
	
	/**
	 * Minimum squared error loss function
	 */
	public static final int MSE = 4;

	/**
	 * root-means squared error loss function
	 */
	public static final int RMSE = 5;
	
	protected DoubleMatrix hBias;
	
	protected DoubleMatrix vBias;
	
	protected DoubleMatrix weights;
	
	protected DoubleMatrix input;
	
	protected DoubleMatrix output;
	
	protected DoubleMatrix activeValue;
	
	protected ReConstructon reConstructon;
	
	protected LearningConfig config;
	
	public LossFunction()
	{
		
	}
	
	public LossFunction(LearningConfig config)
	{
		this.config = config;
	}
	
	public static LossFunction lossFunc(int lossType, LearningConfig config)
	{
		switch (lossType)
		{
		case SQUARED_LOSS:
			return new SquaredLoss(config);
		case MSE:
			return new MSELoss(config);
		case LOGLIKELIHOOD_LOSS:
			return new Loglikelihood(config);
		case RECONSTRUCTION_CROSSENTROPY:
			return new ReConstructCrossEntropy(config);
		case RMSE:
			return new RMSELoss(config);
		}
		return null;
	}
	
	/**
	 * Get loss value
	 * 
	 * @return
	 */
	public abstract double getLossValue();

	public DoubleMatrix getInput()
	{
		return input;
	}

	public void setInput(DoubleMatrix input)
	{
		this.input = input;
	}

	public DoubleMatrix getOutput()
	{
		return output;
	}

	public void setOutput(DoubleMatrix output)
	{
		this.output = output;
	}

	public DoubleMatrix getActiveValue()
	{
		return activeValue;
	}

	public void setActiveValue(DoubleMatrix activeValue)
	{
		this.activeValue = activeValue;
	}

	public LearningConfig getConfig()
	{
		return config;
	}

	public void setConfig(LearningConfig config)
	{
		this.config = config;
	}

	public DoubleMatrix gethBias()
	{
		return hBias;
	}

	public void sethBias(DoubleMatrix hBias)
	{
		this.hBias = hBias;
	}

	public DoubleMatrix getvBias()
	{
		return vBias;
	}

	public void setvBias(DoubleMatrix vBias)
	{
		this.vBias = vBias;
	}

	public DoubleMatrix getWeights()
	{
		return weights;
	}

	public void setWeights(DoubleMatrix weights)
	{
		this.weights = weights;
	}

	public ReConstructon getReConstructon()
	{
		return reConstructon;
	}

	public void setReConstructon(ReConstructon reConstructon)
	{
		this.reConstructon = reConstructon;
	}
	
	
	
}

