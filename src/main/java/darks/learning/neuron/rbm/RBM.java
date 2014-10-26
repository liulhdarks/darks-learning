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

import static darks.learning.common.utils.MatrixHelper.binomial;
import static darks.learning.common.utils.MatrixHelper.columnVariance;
import static darks.learning.common.utils.MatrixHelper.gaussion;
import static darks.learning.common.utils.MatrixHelper.max;
import static darks.learning.common.utils.MatrixHelper.sigmoid;
import static darks.learning.common.utils.MatrixHelper.softmax;
import static darks.learning.common.utils.MatrixHelper.sqrt;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.UnsupervisedLearning;
import darks.learning.exceptions.LearningException;
import darks.learning.neuron.AbstractNeuronNetwork;
import darks.learning.neuron.PropPair;
import darks.learning.neuron.ReConstructon;
import darks.learning.neuron.gradient.GradientComputer;
import darks.learning.neuron.rbm.RBMConfig.LayoutType;

/**
 * Restricted boltzmann machine algorithm
 * 
 * @author Darks.Liu
 *
 */
public class RBM extends AbstractNeuronNetwork implements UnsupervisedLearning, ReConstructon
{
	private static final Logger log = LoggerFactory.getLogger(RBM.class);

	public RBMConfig config = new RBMConfig();

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
		log.info("Complete to train RBM algorithm.cost " + (System.currentTimeMillis() - st) + "ms");
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
	    if(config.visibleType == LayoutType.GAUSSION)
	    {
            this.sigma = columnVariance(input).divi(input.rows);
	    }
		log.info("Initialize RBM visible:" + vSize + " hidden:" + hSize + " weights:" + weights.length);
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
		PropPair hProp1 = sampleHiddenByVisible(input);
		PropPair hPropn = hProp1;
		PropPair vPropn = null;
		int gibbsCount = config.gibbsCount;
		for (int i = 0; i < gibbsCount; i++)
		{
		    vPropn = sampleVisibleByHidden(hPropn.sample);
		    hPropn = sampleHiddenByVisible(vPropn.sample);
		}
		DoubleMatrix wGradient = input.transpose().mmul(hProp1.prob)
		            .sub(vPropn.sample.transpose().mmul(hPropn.prob));
		DoubleMatrix hGradient = hProp1.sample.sub(hPropn.prob).columnMeans();
        DoubleMatrix vGradient = input.sub(vPropn.sample).columnMeans();
       
        gradComputer.computeGradient(wGradient, vGradient, hGradient);
		return gradComputer;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public PropPair sampleHiddenByVisible(DoubleMatrix v)
	{
		DoubleMatrix h1Prob = propForward(v);
		DoubleMatrix h1Sample = null;
		switch (config.hiddenType)
		{
		case LayoutType.BINARY:
			h1Sample = binomial(h1Prob, config.randomFunction);
			break;
		case LayoutType.GAUSSION:
            this.hiddenSigma = columnVariance(h1Prob);
            h1Sample =  h1Prob.addi(gaussion(h1Prob, this.hiddenSigma));
			break;
		case LayoutType.RECTIFIED:
			DoubleMatrix sigProb = sigmoid(h1Prob);
			DoubleMatrix sqrtSig = sqrt(sigProb);
			h1Sample = h1Prob.addi(gaussion(h1Prob, 1).mul(sqrtSig));
			max(0.0, h1Sample);
			break;
		case LayoutType.SOFTMAX:
			h1Sample = softmax(h1Prob);
			break;
		default:
			break;
		}
		return new PropPair(h1Prob, h1Sample);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public PropPair sampleVisibleByHidden(DoubleMatrix h)
	{
	    DoubleMatrix v2Prob = propBackward(h);
	    DoubleMatrix v2Sample = null;
		switch (config.visibleType)
		{
		case LayoutType.BINARY:
            v2Sample = binomial(v2Prob, config.randomFunction);
            break;
		case LayoutType.GAUSSION:
			v2Sample = v2Prob.add(DoubleMatrix.randn(v2Prob.rows, v2Prob.columns));
			break;
		case LayoutType.SOFTMAX:
			v2Sample = softmax(v2Prob);
			break;
		}
		return new PropPair(v2Prob, v2Sample);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix propForward(DoubleMatrix v)
	{
		if(config.visibleType == LayoutType.GAUSSION)
		{
            this.sigma = columnVariance(v).divi(vInput.rows);
		}
		if (v.isColumnVector())
		{
		    v = v.transpose();
		}
		DoubleMatrix preProb = v.mmul(weights);
		if (config.concatBias)
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
			preProb.addi(preProb.add(DoubleMatrix.randn(preProb.rows, preProb.columns)));
            return preProb;
            
		case LayoutType.RECTIFIED:
			return max(0, preProb);
			
		case LayoutType.SOFTMAX:
			return softmax(preProb);
		default:
			throw new LearningException("RBM's hidden type " + config.hiddenType + " is invalid.");
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix propBackward(DoubleMatrix h)
    {
        if (h.isColumnVector())
        {
            h = h.transpose();
        }
        DoubleMatrix preProb = h.mmul(weights.transpose());
        if (config.concatBias)
        {
            preProb = DoubleMatrix.concatHorizontally(preProb, vBias);
        }
        else
        {
            preProb.addiRowVector(vBias);
        }
        switch (config.visibleType)
        {
        case LayoutType.BINARY:
            return sigmoid(preProb);
            
        case LayoutType.GAUSSION:
        	preProb.addi(gaussion(preProb, 1.0));
            return preProb;
            
        case LayoutType.RECTIFIED:

            return null;
        case LayoutType.SOFTMAX:
			return softmax(preProb);
			
        default:
            throw new LearningException("RBM's visible type " + config.hiddenType + " is invalid.");
        }
    }
	
	

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix reconstruct(DoubleMatrix input)
	{
		return propBackward(propForward(input));
	}
}
