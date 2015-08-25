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
package darks.learning.neuron.sda;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.SupervisedLearning;
import darks.learning.classifier.regression.LogisticRegression;
import darks.learning.classifier.regression.Regression;
import darks.learning.classifier.regression.SoftmaxRegression;
import darks.learning.neuron.PretrainFinetuneLearning;
import darks.learning.neuron.da.DenoisingAutoEncoder;
import darks.learning.neuron.sda.SdaConfig.FinetuneType;

/**
 * Deep belief network
 * 
 * @author Darks.Liu
 *
 */
public class StackedDenosingAutoEncoder extends PretrainFinetuneLearning
{
	
	private static Logger log = LoggerFactory.getLogger(StackedDenosingAutoEncoder.class);
	
	public SdaConfig config = new SdaConfig();
	
	DenoisingAutoEncoder[] hiddenLayers;
	
	SupervisedLearning outputLayer;
	
	public StackedDenosingAutoEncoder()
	{
		
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void trainBatch(DoubleMatrix input, DoubleMatrix output)
	{
		buildConfig();
		DoubleMatrix preInput = pretrain(input);
		if (preInput != null)
		{
			finetune(preInput, output);
		}
	}
	
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix pretrain(DoubleMatrix initInput)
	{
		DoubleMatrix input = initInput;
		for (int i = 0; i < hiddenLayers.length; i++)
		{
			DenoisingAutoEncoder hiddenLayer = hiddenLayers[i];
			if (log.isDebugEnabled())
			{
				log.debug("Start to train " + i + " layout DenoisingAutoEncoder algorithm.");
			}
			hiddenLayer.train(input);
			if (config.useSample)
			{
				input = hiddenLayer.sampleHiddenByVisible(input).getSample();
			}
			else
			{
				input = hiddenLayer.propForward(input);
			}
		}
		return input;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void finetune(DoubleMatrix input, DoubleMatrix output)
	{
		if (log.isDebugEnabled())
		{
			log.debug("Start to train finetune layout." + outputLayer);
		}
		outputLayer.trainBatch(input, output);
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix predict(DoubleMatrix initInput)
	{
		buildConfig();
		DoubleMatrix input = initInput;
		for (int i = 0; i < hiddenLayers.length; i++)
		{
			DenoisingAutoEncoder hiddenLayer = hiddenLayers[i];
			if (config.useSample)
			{
				input = hiddenLayer.sampleHiddenByVisible(input).getSample();
			}
			else
			{
				input = hiddenLayer.propForward(input);
			}
		}
		return outputLayer.predict(input);
	}

	
	private void buildConfig()
	{
		if (hiddenLayers != null && outputLayer != null)
		{
			return;
		}
		config.checkConfigValid();
		int hiddenCount = config.hiddenLayouts.length;
		hiddenLayers = new DenoisingAutoEncoder[hiddenCount];
		for (int i = 0; i < hiddenCount; i++)
		{
			DenoisingAutoEncoder da = buildHiddenLayer(i);
			hiddenLayers[i] = da;
		}
		outputLayer = buildOutputLayer();
	}
	
	private DenoisingAutoEncoder buildHiddenLayer(int layer)
	{
		DenoisingAutoEncoder da = new DenoisingAutoEncoder();
		da.config.setHiddenSize(config.getHiddenLayouts(layer))
					.setLearnRate(config.getHiddensLearnRate(layer))
					.setLossType(config.getHiddenLossTypes(layer))
					.setMaxIterateCount(config.getHiddensIterateCount(layer))
					.setMomentum(config.getMomentums(layer))
					.setNormalized(config.normalized)
					.setOptimizeType(config.getOptimizeTypes(layer))
					.setUseAdaGrad(config.getUseAdaGrads(layer))
					.setRandomFunction(config.randomFunction)
					.setCorruptionLevel(config.getCorruptionLevel());
		
		return da;
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
