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
package darks.learning.regression;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Logistic regression
 * 
 * @author Darks.Liu
 *
 */
public class LogisticRegression extends Regression
{
	
	private static Logger log = LoggerFactory.getLogger(LogisticRegression.class);
	
	DoubleMatrix weight = null;
	
	DoubleMatrix bias = null;
	
	double learnRate;
	
	public LogisticRegression()
	{
		
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void train(DoubleMatrix input, DoubleMatrix output)
	{
		double startLearnRate = config.learnRate;
		learnRate = startLearnRate;
		initWeight(input, output);
		int iterCount = config.maxIteratorCount;
		for (int i = 1; i <= iterCount; i++)
		{
			learnRate = startLearnRate + (1 / (double)(i + 3));
			iterator(input, output);
			double likelihook = calcuateLogLikelihook(input, output);
			if (log.isDebugEnabled() && i % 1000 == 0)
			{
				log.debug("Iterator:" + i + " cost:" + likelihook + " lr:" + learnRate);
			}
		}
	}
	
	private void initWeight(DoubleMatrix input, DoubleMatrix output)
	{
		weight = DoubleMatrix.rand(input.columns, output.columns);//DoubleMatrix.ones(input.columns, 1);
		bias = DoubleMatrix.zeros(1, output.columns);
	}
	
	private void iterator(DoubleMatrix input, DoubleMatrix output)
	{
		if (config.randomGradient)
		{
			randomGradientDescent(input, output);
		}
		else
		{
			gradientDescent(input, output);
		}
	}
	
	private void gradientDescent(DoubleMatrix input, DoubleMatrix output)
	{
		DoubleMatrix f = config.activateFunction.activate(input.mmul(weight).addRowVector(bias));
		DoubleMatrix error = f.sub(output);
		DoubleMatrix theta = error.mul(learnRate);
		DoubleMatrix delta = input.transpose().mmul(theta);
		theta = theta.columnSums();
		if (config.normalized)
		{
			delta.divi(input.rows);
			theta.divi(input.rows);
		}
		weight.subi(delta);
		bias.subi(theta);
	}
	
	private void randomGradientDescent(DoubleMatrix input, DoubleMatrix output)
	{
		int rows = input.rows;
		for (int i = 0; i < rows; i++)
		{
			int index = config.randomFunction.randInt(rows);
			DoubleMatrix rowMatrix = input.getRow(index);
			gradientDescent(rowMatrix, output.getRow(index));
		}
	}
	
	private double calcuateLogLikelihook(DoubleMatrix input, DoubleMatrix output)
	{
		DoubleMatrix f = config.activateFunction.activate(input.mmul(weight));
		config.lossFunction.setActiveValue(f);
		config.lossFunction.setInput(input);
		config.lossFunction.setOutput(output);
		return config.lossFunction.getLossValue();
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix predict(DoubleMatrix input)
	{
		return config.activateFunction.activate(input.mmul(weight));
	}

}
