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
import static darks.learning.common.utils.MatrixHelper.oneMinus;

import org.jblas.DoubleMatrix;

import darks.learning.LearningConfig;

/**
 * Loss function
 * 
 * @author Darks.Liu
 *
 */
public class LossFunction
{
	
	/**
	 * Squared loss function
	 */
	public static final int SQUARED_LOSS = 1;
	
	/**
	 * Negative log likelihood loss function.
	 * y*log(h(x))+(1-y)*log(1-h(x))
	 */
	public static final int NEG_LOGLIKELIHOOD_LOSS = 2;
	
	private int lossType = 0;
	
	private DoubleMatrix input;
	
	private DoubleMatrix output;
	
	private DoubleMatrix activeValue;
	
	private LearningConfig config;
	
	private LossFunction(int lossType, LearningConfig config)
	{
		this.lossType = lossType;
		this.config = config;
	}
	
	private LossFunction(int lossType, DoubleMatrix input, 
			DoubleMatrix output, DoubleMatrix activeValue, LearningConfig config)
	{
		this.lossType = lossType;
		this.input = input;
		this.output = output;
		this.activeValue = activeValue;
		this.config = config;
	}
	
	public static LossFunction lossFunc(int lossType, LearningConfig config)
	{
		return new LossFunction(lossType, config);
	}
	
	public static LossFunction lossFunc(int lossType, DoubleMatrix input, 
			DoubleMatrix output, DoubleMatrix activeVal, LearningConfig config)
	{
		return new LossFunction(lossType, input, output, activeVal, config);
	}
	
	public double getLossValue()
	{
		switch (lossType)
		{
		case SQUARED_LOSS:
			return squaredLoss();
		case NEG_LOGLIKELIHOOD_LOSS:
			return -logLikelihoodLoss();
		default:
			return 0;
		}
	}
	
	private double squaredLoss()
	{
		
		return 0;
	}
	
	
	private double logLikelihoodLoss()
	{
		double likelihood = output.mul(log(activeValue)).add(
                oneMinus(output).mul(log(oneMinus(activeValue)))).
                columnSums().mean();
		if (config.normalized)
		{
			likelihood /= input.rows;
		}
		return likelihood;
	}

	public int getLossType()
	{
		return lossType;
	}

	public void setLossType(int lossType)
	{
		this.lossType = lossType;
	}

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
	
}

