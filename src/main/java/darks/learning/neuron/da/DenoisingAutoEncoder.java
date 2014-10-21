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
package darks.learning.neuron.da;

import static darks.learning.common.utils.MatrixHelper.oneMinus;
import static darks.learning.common.utils.MatrixHelper.binomial;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.UnsupervisedLearning;
import darks.learning.neuron.AbstractNeuronNetwork;
import darks.learning.neuron.PropPair;
import darks.learning.neuron.ReConstructon;
import darks.learning.neuron.gradient.GradientComputer;

/**
 * Denoising auto encoder
 * 
 * @author Darks.Liu
 *
 */
public class DenoisingAutoEncoder extends AbstractNeuronNetwork implements UnsupervisedLearning, ReConstructon
{
	private static final Logger log = LoggerFactory.getLogger(DenoisingAutoEncoder.class);

	public AutoEncoderConfig config = new AutoEncoderConfig();
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public void train(DoubleMatrix input)
	{
		long st = System.currentTimeMillis();
		initialize(input);
		if (optimizer != null)
		{
			optimizer.optimize();
		}
		log.info("Complete to train DenoisingAutoEncoder algorithm.cost " + (System.currentTimeMillis() - st) + "ms");
	}
	
	private void initialize(DoubleMatrix input)
	{
		initialize(config);
		vInput = input;
		int vSize = config.visibleSize <= 0 ? input.columns : config.visibleSize;
		int hSize = config.hiddenSize;
		weights = DoubleMatrix.randn(vSize, hSize);
		vBias = DoubleMatrix.rand(vSize);
		hBias = DoubleMatrix.zeros(hSize);
		fillBias(input);
		log.info("Initialize DenoisingAutoEncoder visible:" + vSize + " hidden:" + hSize + " weights:" + weights.length);
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

	/**
	 * {@inheritDoc}
	 */
	@Override
	public GradientComputer getGradient(DoubleMatrix input)
	{
		gradComputer.setBatchSize(input.rows);
		gradComputer.setWeights(weights);
		gradComputer.sethBias(hBias);
		gradComputer.setvBias(vBias);

		DoubleMatrix corruptX = getCorruptInput(input);
		DoubleMatrix y = propForward(corruptX);
		DoubleMatrix v2 = propBackward(y);
		DoubleMatrix visibleLoss = input.sub(v2);
		DoubleMatrix hiddenLoss = visibleLoss.mmul(weights).mul(y).mul(oneMinus(y));
		
		DoubleMatrix wGradient = corruptX.transpose().mmul(hiddenLoss).add(visibleLoss.transpose().mmul(y));
		DoubleMatrix vGradient = visibleLoss.columnMeans();
		DoubleMatrix hGradient = hiddenLoss.columnMeans();
       
        gradComputer.computeGradient(wGradient, vGradient, hGradient);
		return gradComputer;
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix reconstruct(DoubleMatrix input)
	{
		return propBackward(propForward(input));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix propForward(DoubleMatrix v)
	{
		DoubleMatrix preSig = null;
		preSig = v.mmul(weights).addiRowVector(hBias);
		return config.activateFunction.activate(preSig);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix propBackward(DoubleMatrix h)
	{
		DoubleMatrix preSig = null;
		preSig = h.mmul(weights.transpose()).addiRowVector(vBias);
		return config.activateFunction.activate(preSig);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public PropPair sampleHiddenByVisible(DoubleMatrix v)
	{
		DoubleMatrix hidden = propForward(v);
		return new PropPair(hidden, hidden);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public PropPair sampleVisibleByHidden(DoubleMatrix h)
	{
		DoubleMatrix visible = propBackward(h);
		return new PropPair(visible, visible);
	}
	
	private DoubleMatrix getCorruptInput(DoubleMatrix input)
	{
		DoubleMatrix mask = new DoubleMatrix(input.rows, input.columns);
		for (int i = 0; i < input.rows; i++)
		{
			for (int j = 0; j < input.columns; j++)
			{
				mask.put(i, j, binomial(1 - config.corruptionLevel, config.randomFunction));
			}
		}
		return mask.mul(input);
	}
	
	
}
