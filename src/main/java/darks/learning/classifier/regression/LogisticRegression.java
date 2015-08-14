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
package darks.learning.classifier.regression;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.neuron.activate.Activations;

/**
 * Logistic regression
 * 
 * @author Darks.Liu
 *
 */
public class LogisticRegression extends Regression
{
	
	private static Logger log = LoggerFactory.getLogger(LogisticRegression.class);
	
	public LogisticRegression()
	{
		config.setActivateFunction(Activations.sigmoid());
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void trainBatch(DoubleMatrix input, DoubleMatrix output)
	{
		log.info("Training logistic regression.");
		if (!output.isScalar())
		{
			output = scalarMatrix(output);
		}
		double startLearnRate = config.learnRate;
		learnRate = startLearnRate;
		initWeight(input, output);
		int iterCount = config.maxIteratorCount;
		for (int i = 1; i <= iterCount; i++)
		{
			if (!config.useAdaGrad)
			{
				learnRate = startLearnRate + (1 / (double)(i + 3));
			}
			iterator(input, output);
			double costValue = calcuateLossValue(input, output);
			if (log.isDebugEnabled() && i % 1000 == 0)
			{
				log.debug("Iterator:" + i + " cost:" + costValue + " lr:" + (config.useAdaGrad ? "Adagrad" : learnRate));
			}
		}
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public double train(int iterateNumber, DoubleMatrix input, DoubleMatrix output)
	{
		if (!output.isScalar())
		{
			output = scalarMatrix(output);
		}
		initWeight(input, output);
		if (!config.useAdaGrad)
		{
			double startLearnRate = config.learnRate;
			learnRate = startLearnRate + (1 / (double)(iterateNumber + 3));
		}
		iterator(input, output);
		return calcuateLossValue(input, output);
	}

	private DoubleMatrix scalarMatrix(DoubleMatrix output)
	{
		DoubleMatrix result = new DoubleMatrix(output.rows);
		for (int i = 0; i < output.rows; i++)
		{
			result.put(i, SimpleBlas.iamax(output.getRow(i)));
		}
		return result;
	}
	
	private void initWeight(DoubleMatrix input, DoubleMatrix output)
	{
		if (weights == null && bias == null)
		{
			weights = DoubleMatrix.ones(input.columns, 1);
			bias = DoubleMatrix.ones(1, 1);
		}
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
	
	private double calcuateLossValue(DoubleMatrix input, DoubleMatrix output)
	{
		DoubleMatrix f = config.activateFunction.activate(input.mmul(weights));
		config.lossFunction.setActiveValue(f);
		config.lossFunction.setInput(input);
		config.lossFunction.setOutput(output);
		config.lossFunction.setWeights(weights);
		config.lossFunction.sethBias(bias);
		return -config.lossFunction.getLossValue();
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix predict(DoubleMatrix input)
	{
		return config.activateFunction.activate(input.mmul(weights));
	}

}
