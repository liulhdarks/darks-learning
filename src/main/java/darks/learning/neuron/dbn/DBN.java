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
package darks.learning.neuron.dbn;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.SupervisedLearning;
import darks.learning.neuron.dbn.DBNConfig.FinetuneType;
import darks.learning.neuron.rbm.RBM;
import darks.learning.regression.LogisticRegression;
import darks.learning.regression.Regression;
import darks.learning.regression.SoftmaxRegression;

/**
 * Deep belief network
 * 
 * @author Darks.Liu
 *
 */
public class DBN implements SupervisedLearning
{
	
	private static Logger log = LoggerFactory.getLogger(DBN.class);
	
	public DBNConfig config = new DBNConfig();
	
	RBM[] rbmLayers;
	
	SupervisedLearning outputLayer;
	
	public DBN()
	{
		
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void train(DoubleMatrix input, DoubleMatrix output)
	{
		buildConfig();
		DoubleMatrix preInput = pretrain(input);
		if (preInput != null)
		{
			finetune(preInput, output);
		}
	}
	
	private DoubleMatrix pretrain(DoubleMatrix initInput)
	{
		DoubleMatrix input = initInput;
		for (int i = 0; i < rbmLayers.length; i++)
		{
			RBM rbmLayer = rbmLayers[i];
			if (log.isDebugEnabled())
			{
				log.debug("Start to train " + i + " layout RBM algorithm.");
			}
			rbmLayer.train(input);
			if (config.useSample)
			{
				input = rbmLayer.sampleHiddenByVisible(input).getSample();
			}
			else
			{
				input = rbmLayer.propForward(input);
			}
		}
		return input;
	}
	
	private void finetune(DoubleMatrix input, DoubleMatrix output)
	{
		if (log.isDebugEnabled())
		{
			log.debug("Start to train finetune layout." + outputLayer);
		}
		outputLayer.train(input, output);
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix predict(DoubleMatrix initInput)
	{
		buildConfig();
		DoubleMatrix input = initInput;
		for (int i = 0; i < rbmLayers.length; i++)
		{
			RBM rbmLayer = rbmLayers[i];
			if (config.useSample)
			{
				input = rbmLayer.sampleHiddenByVisible(input).getSample();
			}
			else
			{
				input = rbmLayer.propForward(input);
			}
		}
		return outputLayer.predict(input);
	}

	
	private void buildConfig()
	{
		if (rbmLayers != null && outputLayer != null)
		{
			return;
		}
		config.checkConfigValid();
		int hiddenCount = config.hiddenLayouts.length;
		rbmLayers = new RBM[hiddenCount];
		for (int i = 0; i < hiddenCount; i++)
		{
			RBM rbm = buildRBMLayer(i);
			rbmLayers[i] = rbm;
		}
		outputLayer = buildOutputLayer();
	}
	
	private RBM buildRBMLayer(int layer)
	{
		RBM rbm = new RBM();
		rbm.config.setHiddenSize(config.getHiddenLayouts(layer))
					.setGibbsCount(config.getGibbsCounts(layer))
					.setHiddenType(config.getHiddenType())
					.setLearnRate(config.getHiddensLearnRate(layer))
					.setLossType(config.getHiddenLossTypes(layer))
					.setMaxIterateCount(config.getHiddensIterateCount(layer))
					.setMomentum(config.getMomentums(layer))
					.setNormalized(config.normalized)
					.setOptimizeType(config.getOptimizeTypes(layer))
					.setUseAdaGrad(config.getUseAdaGrads(layer))
					.setRandomFunction(config.randomFunction)
					.setVisibleType(config.getVisibleType());
		
		return rbm;
	}
	
	private SupervisedLearning buildOutputLayer()
	{
		if (config.fineTuneLayer != null)
		{
			return config.fineTuneLayer;
		}
		SupervisedLearning result = null;
		if (config.finetuneType == FinetuneType.SOFTMAX || config.finetuneType == FinetuneType.LOGISTIC)
		{
			Regression regress = null;
			if (config.finetuneType == FinetuneType.SOFTMAX)
			{
				regress = new SoftmaxRegression();
			}
			else if (config.finetuneType == FinetuneType.LOGISTIC)
			{
				regress = new LogisticRegression();
			}
			regress.config.setLearnRate(config.getFineTuneLearnRate())
						.setMaxIteratorCount(config.getFineTuneIterateCount())
						.setRandomGradient(false)
						.setNormalized(config.normalized)
						.setActivateFunction(config.getFineTuneActivation())
						.setLossType(config.getFineTuneLossType())
						.setRandomFunction(config.randomFunction)
						.setUseAdaGrad(config.isFineTuneUseAdaGrads());
			result = regress;
		}
		return result;
	}
}
