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
package darks.learning.neuron.rbm;

import static darks.learning.common.utils.MatrixHelper.sigmoid;
import static darks.learning.common.utils.MatrixHelper.softmax;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.MatrixHelper;
import darks.learning.exceptions.LearningException;
import darks.learning.neuron.UnsupervisedLearning;
import darks.learning.neuron.rbm.RBMConfig.LayoutType;

/**
 * Restricted boltzmann machine algorithm
 * 
 * @author Darks.Liu
 *
 */
public class RBM implements UnsupervisedLearning
{
	private static final Logger log = LoggerFactory.getLogger(RBM.class);

	public RBMConfig config = new RBMConfig();
	
	DoubleMatrix weights;
	
	DoubleMatrix vBias;
	
	DoubleMatrix hBias;
	
	DoubleMatrix hInput;
	
	DoubleMatrix vInput;
	
	DoubleMatrix initVInput;
	
	long startTime;
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public void train(DoubleMatrix input)
	{
		initialize(input);
		int iterCount = config.maxIterateCount;
		int epoch = 0;
		while (iterCount == 0 || epoch < iterCount)
		{
			gibbsSampling(input);
			if (!checkIterateTime())
			{
				break;
			}
		}
	}
	
	private void initialize(DoubleMatrix input)
	{
		startTime = System.currentTimeMillis();
		initVInput = input;
		int vSize = config.visibleSize <= 0 ? input.columns : config.visibleSize;
		int hSize = config.hiddenSize;
		weights = DoubleMatrix.randn(vSize, hSize);
		vBias = DoubleMatrix.rand(vSize);
		hBias = DoubleMatrix.zeros(hSize);
		fillBias(input);
	}
	
	private void fillBias(DoubleMatrix input)
	{
		int[] sum = new int[input.columns];
		for (int r = 0; r < input.rows; r++)
		{
			DoubleMatrix row = input.getRow(r);
			for (int c = 0; c < row.columns; c++)
			{
				if (config.randomFunction.randDouble() <= row.get(c))
				{
					sum[c]++;
				}
			}
		}
		
		for (int i = 0; i < input.columns; i++)
		{
			if (sum[i] == 0) 
			{
				sum[i] = 1;
			}
			if (sum[i] > 0 && sum[i] != input.rows)
			{
				double p = (double)sum[i] / (double)input.rows;
				vBias.put(i, Math.log(p / (1 - p)));
			}
		}
	}
	
	private void gibbsSampling(DoubleMatrix input)
	{
		PropPair prop1 = sampleHiddenByVisible(input);
		sampleVisibleByHidden();
	}
	
	private PropPair sampleHiddenByVisible(DoubleMatrix v)
	{
		DoubleMatrix h1Prob = propForward(v);
		switch (config.hiddenType)
		{
		case LayoutType.BINARY:
			DoubleMatrix h1Sample = MatrixHelper.binomial(h1Prob, config.randomFunction);
			return new PropPair(h1Prob, h1Sample);
		case LayoutType.GAUSSION:
			
			break;
		case LayoutType.RECTIFIED:
			
			break;
		case LayoutType.SOFTMAX:
			
			break;
		default:
			break;
		}
		return null;
	}
	
	private void sampleVisibleByHidden()
	{
		switch (config.visibleType)
		{
		case LayoutType.BINARY:
			
			break;
		case LayoutType.GAUSSION:
			
			break;
		case LayoutType.RECTIFIED:
			
			break;
		case LayoutType.SOFTMAX:
			
			break;
		}
	}
	
	private DoubleMatrix propForward(DoubleMatrix v)
	{
		DoubleMatrix preProb = v.mmul(weights);
		if (config.convatBias)
		{
			preProb = DoubleMatrix.concatHorizontally(preProb, hBias);
		}
		else
		{
			preProb.addiRowVector(hBias);
		}
		switch (config.hiddenType)
		{
		case LayoutType.BINARY:
			return sigmoid(preProb);
			
		case LayoutType.GAUSSION:

			return null;
		case LayoutType.RECTIFIED:

			return null;
		case LayoutType.SOFTMAX:
			
			return null;
		default:
			throw new LearningException("RBM's hidden type " + config.hiddenType + " is invalid.");
		}
	}
	
	private boolean checkIterateTime()
	{
		if (config.maxIterateTime > 0 
				&& System.currentTimeMillis() - startTime >= config.maxIterateTime)
		{
			return false;
		}
		return true;
	}
	
	
	class PropPair
	{
		DoubleMatrix prob;
		
		DoubleMatrix sample;

		public PropPair(DoubleMatrix prob, DoubleMatrix sample)
		{
			this.prob = prob;
			this.sample = sample;
		}
		
	}
}
