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
import darks.learning.classifier.regression.LogisticRegression;
import darks.learning.classifier.regression.Regression;
import darks.learning.classifier.regression.SoftmaxRegression;
import darks.learning.neuron.PretrainFinetuneLearning;
import darks.learning.neuron.dbn.DBNConfig.FinetuneType;
import darks.learning.neuron.mlp.MultiLayerNeuronNetwork;
import darks.learning.neuron.rbm.RBM;

/**
 * Deep belief network
 * 
 * @author Darks.Liu
 *
 */
public class DBN extends PretrainFinetuneLearning
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
	public void trainBatch(DoubleMatrix input, DoubleMatrix output)
	{
		buildConfig();
		DoubleMatrix preInput = pretrain(input);
		if (preInput != null)
		{
			if (config.finetuneType != FinetuneType.MLP)
				finetune(preInput, output);
			else
				finetune(input, output);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix pretrain(DoubleMatrix initInput)
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
		buildFinetuneAfterPretrain();
		outputLayer.trainBatch(input, output);
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix predict(DoubleMatrix initInput)
	{
		if (rbmLayers == null || outputLayer == null)
		{
			return null;
		}
		DoubleMatrix input = initInput;
		if (config.finetuneType != FinetuneType.MLP)
		{
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
		if (config.finetuneType != FinetuneType.MLP)
		{
			outputLayer = buildOutputLayer();
		}
	}
	
	private void buildFinetuneAfterPretrain()
	{
		if (outputLayer == null && config.finetuneType != FinetuneType.MLP)
			return;
		int inputLayerSize = rbmLayers[0].config.getVisibleSize();
		MultiLayerNeuronNetwork mlp = new MultiLayerNeuronNetwork();
        mlp.config.setUseAdaGrad(config.isFineTuneUseAdaGrads())
                    .setLearnRate(config.getFineTuneLearnRate())
                    .setMomentum(0.7)
                    .setMaxIterateCount(config.fineTuneIterateCount)
					.setLossType(config.getFineTuneLossType())
					.setInputLayerSize(inputLayerSize)
                    .setOutputLayerSize(config.fineTuneOutputLayerSize)
                    .setRandomFunction(config.randomFunction)
                    .setUseRegularization(false)
                    .setHiddenLayouts(config.getHiddenLayouts());
//                    .setUseRegularization(true)
//                    .setL2(0.1);
        mlp.initialize();
        for (int i = 0; i < rbmLayers.length; i++)
		{
			RBM rbmLayer = rbmLayers[i];
			mlp.setHiddenLayerParams(i, rbmLayer);
		}
		outputLayer = mlp;
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
						.setNormalized(config.normalized)
						.setActivateFunction(config.getFineTuneActivation())
						.setLossType(config.getFineTuneLossType())
						.setRandomFunction(config.randomFunction)
						.setUseAdaGrad(config.isFineTuneUseAdaGrads());
			result = regress;
		}
		else if (config.finetuneType == FinetuneType.MLP_CONCAT)
		{
			MultiLayerNeuronNetwork mlp = new MultiLayerNeuronNetwork();
	        mlp.config.setUseAdaGrad(config.isFineTuneUseAdaGrads())
	                    .setLearnRate(config.getFineTuneLearnRate())
	                    .setMomentum(0.7)
	                    .setMaxIterateCount(config.fineTuneIterateCount)
						.setLossType(config.getFineTuneLossType())
						.setInputLayerSize(config.hiddenLayouts[config.hiddenLayouts.length - 1])
	                    .setOutputLayerSize(config.fineTuneOutputLayerSize)
	                    .setRandomFunction(config.randomFunction)
	                    .setUseRegularization(false);
	        if (config.fineTuneHiddenLayouts != null)
	        	mlp.config.setHiddenLayouts(config.fineTuneHiddenLayouts);
	        else
	        {
	        	int lastLaysize = config.hiddenLayouts[config.hiddenLayouts.length - 1];
	        	mlp.config.setHiddenLayouts(new int[]{lastLaysize / 2});
	        }
			result = mlp;
		}
		return result;
	}
}
