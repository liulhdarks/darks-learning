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

import darks.learning.common.utils.MatrixHelper;

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
		initWeight(input);
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
	
	private void initWeight(DoubleMatrix input)
	{
		if (config.randomGradient)
		{
			weight = DoubleMatrix.ones(input.columns);
		}
		else
		{
			weight = DoubleMatrix.ones(input.columns, 1);
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
	
	private void gradientDescent(DoubleMatrix input, DoubleMatrix output)
	{
		DoubleMatrix f = MatrixHelper.sigmoid(input.mmul(weight));
		DoubleMatrix error = f.sub(output);
		weight.subi(input.transpose().mmul(error).mul(learnRate).div(input.rows));
	}
	
	private void randomGradientDescent(DoubleMatrix input, DoubleMatrix output)
	{
		
	}
	
	private double calcuateLogLikelihook(DoubleMatrix input, DoubleMatrix output)
	{
		int m = output.rows;
		DoubleMatrix ones = DoubleMatrix.ones(output.rows, output.columns);
		DoubleMatrix f = MatrixHelper.sigmoid(input.mmul(weight));
		DoubleMatrix a = output.mmul(MatrixHelper.log(f).transpose());
		DoubleMatrix b = ones.sub(output).mmul(MatrixHelper.log(ones.sub(f)).transpose());
		DoubleMatrix ret = a.add(b);
		return ret.sum() * (1.0 / m);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix predict(DoubleMatrix input)
	{
		return MatrixHelper.sigmoid(input.mmul(weight));
	}

}
