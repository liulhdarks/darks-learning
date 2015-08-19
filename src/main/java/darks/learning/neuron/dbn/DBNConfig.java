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

import darks.learning.LearningConfig;
import darks.learning.SupervisedLearning;
import darks.learning.common.rand.RandomFunction;
import darks.learning.exceptions.ConfigException;
import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.activate.ActivateFunction;
import darks.learning.neuron.activate.Activations;
import darks.learning.neuron.rbm.RBMConfig.LayoutType;
import darks.learning.optimize.LearningOptimizer.OptimizeType;

/**
 * DBN Configuration
 * 
 * @author Darks.Liu
 *
 */
public class DBNConfig extends LearningConfig
{

	public enum FinetuneType
	{
		SOFTMAX, LOGISTIC, MLP, MLP_CONCAT
	}
	
	double fineTuneLearnRate = 1e-3;
	
	int fineTuneIterateCount = 10000;
	
	ActivateFunction fineTuneActivation;
	
	int fineTuneLossType = LossFunction.LOGLIKELIHOOD_LOSS; 
	
	FinetuneType finetuneType = FinetuneType.SOFTMAX;
	
	boolean fineTuneUseAdaGrads = false;
	
	int[] fineTuneHiddenLayouts;
	
	int fineTuneOutputLayerSize;
	
	SupervisedLearning fineTuneLayer;
	
	double[] hiddensLearnRate;
	
	int[] hiddensIterateCount;
	
	int[] hiddenLayouts;
	
	int[] hiddenLossTypes;
	
	int[] gibbsCounts;
	
	boolean[] useAdaGrads;
	
	OptimizeType[] optimizeTypes;
	
	double[] momentums;
	
	int visibleType = LayoutType.BINARY;
	
	int hiddenType = LayoutType.BINARY;
	
	boolean useSample = false;
	
	public DBNConfig()
	{
		
	}
	
	public void checkConfigValid()
	{
		if (hiddenLayouts == null)
		{
			throw new ConfigException("DBN's hidden layouts cannot be null");
		}
	}

	public DBNConfig setFineTuneLearnRate(double fineTuneLearnRate)
	{
		this.fineTuneLearnRate = fineTuneLearnRate;
		return this;
	}

	public DBNConfig setFineTuneIterateCount(int fineTuneIterateCount)
	{
		this.fineTuneIterateCount = fineTuneIterateCount;
		return this;
	}

	public DBNConfig setFineTuneActivation(ActivateFunction fineTuneActivation)
	{
		this.fineTuneActivation = fineTuneActivation;
		return this;
	}

	public DBNConfig setFineTuneLossType(int fineTuneLossType)
	{
		this.fineTuneLossType = fineTuneLossType;
		return this;
	}

	public DBNConfig setHiddensLearnRate(double hiddensLearnRate)
	{
		this.hiddensLearnRate = new double[]{hiddensLearnRate};
		return this;
	}

	public DBNConfig setHiddensLearnRate(double[] hiddensLearnRate)
	{
		this.hiddensLearnRate = hiddensLearnRate;
		return this;
	}

	public DBNConfig setHiddensIterateCount(int hiddensIterateCount)
	{
		this.hiddensIterateCount = new int[]{hiddensIterateCount};
		return this;
	}

	public DBNConfig setHiddensIterateCount(int[] hiddensIterateCount)
	{
		this.hiddensIterateCount = hiddensIterateCount;
		return this;
	}

	public DBNConfig setHiddenLayouts(int hiddenLayouts)
	{
		this.hiddenLayouts = new int[]{hiddenLayouts};
		return this;
	}

	public DBNConfig setHiddenLayouts(int[] hiddenLayouts)
	{
		this.hiddenLayouts = hiddenLayouts;
		return this;
	}

	public DBNConfig setHiddenLossTypes(int hiddenLossTypes)
	{
		this.hiddenLossTypes = new int[]{hiddenLossTypes};
		return this;
	}

	public DBNConfig setHiddenLossTypes(int[] hiddenLossTypes)
	{
		this.hiddenLossTypes = hiddenLossTypes;
		return this;
	}

	public DBNConfig setGibbsCounts(int gibbsCounts)
	{
		this.gibbsCounts = new int[]{gibbsCounts};
		return this;
	}

	public DBNConfig setGibbsCounts(int[] gibbsCounts)
	{
		this.gibbsCounts = gibbsCounts;
		return this;
	}

	public DBNConfig setUseAdaGrads(boolean useAdaGrads)
	{
		this.useAdaGrads = new boolean[]{useAdaGrads};
		return this;
	}

	public DBNConfig setUseAdaGrads(boolean[] useAdaGrads)
	{
		this.useAdaGrads = useAdaGrads;
		return this;
	}

	public DBNConfig setOptimizeTypes(OptimizeType optimizeTypes)
	{
		this.optimizeTypes = new OptimizeType[]{optimizeTypes};
		return this;
	}

	public DBNConfig setOptimizeTypes(OptimizeType[] optimizeTypes)
	{
		this.optimizeTypes = optimizeTypes;
		return this;
	}

	public DBNConfig setMomentums(double momentums)
	{
		this.momentums = new double[]{momentums};
		return this;
	}

	public DBNConfig setMomentums(double[] momentums)
	{
		this.momentums = momentums;
		return this;
	}


	public DBNConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public DBNConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}
	
	public DBNConfig setVisibleType(int visibleType)
	{
		this.visibleType = visibleType;
		return this;
	}

	public DBNConfig setHiddenType(int hiddenType)
	{
		this.hiddenType = hiddenType;
		return this;
	}

	public DBNConfig setHiddenLayoutType(int layoutType)
	{
		this.visibleType = layoutType;
		this.hiddenType = layoutType;
		return this;
	}

	public DBNConfig setUseSample(boolean useSample)
	{
		this.useSample = useSample;
		return this;
	}
	
	public DBNConfig setFinetuneType(FinetuneType finetuneType)
	{
		this.finetuneType = finetuneType;
		return this;
	}

	public DBNConfig setFineTuneLayer(SupervisedLearning fineTuneLayer)
	{
		this.fineTuneLayer = fineTuneLayer;
		return this;
	}

	public DBNConfig setFineTuneUseAdaGrads(boolean fineTuneUseAdaGrads)
	{
		this.fineTuneUseAdaGrads = fineTuneUseAdaGrads;
		return this;
	}

	public DBNConfig setFineTuneHiddenLayouts(int[] fineTuneHiddenLayouts)
	{
		this.fineTuneHiddenLayouts = fineTuneHiddenLayouts;
		return this;
	}

	public DBNConfig setFineTuneOutputLayerSize(int fineTuneOutputLayerSize)
	{
		this.fineTuneOutputLayerSize = fineTuneOutputLayerSize;
		return this;
	}

	/*-------------------------------------------------------------------------------------------------
	 * Get function 
	 *-------------------------------------------------------------------------------------------------
	 */
	public double getFineTuneLearnRate()
	{
		return fineTuneLearnRate;
	}

	public int getFineTuneIterateCount()
	{
		return fineTuneIterateCount;
	}

	public ActivateFunction getFineTuneActivation()
	{
		return fineTuneActivation == null ? Activations.softmax() : fineTuneActivation;
	}

	public int getFineTuneLossType()
	{
		return fineTuneLossType;
	}

	public double getHiddensLearnRate(int layer)
	{
		return hiddensLearnRate == null ? 0.001 : 
					(hiddensLearnRate.length <= layer ? hiddensLearnRate[0] : hiddensLearnRate[layer]);
	}

	public int getHiddensIterateCount(int layer)
	{
		return hiddensIterateCount == null ? 10000 : 
					(hiddensIterateCount.length <= layer ? hiddensIterateCount[0] : hiddensIterateCount[layer]);
	}

	public int getHiddenLayouts(int layer)
	{
		return hiddenLayouts.length <= layer ? hiddenLayouts[0] : hiddenLayouts[layer];
	}

	public int[] getHiddenLayouts()
	{
		return hiddenLayouts;
	}

	public int getHiddenLossTypes(int layer)
	{
		return hiddenLossTypes == null ? LossFunction.RECONSTRUCTION_CROSSENTROPY : 
				(hiddenLossTypes.length <= layer ? hiddenLossTypes[0] : hiddenLossTypes[layer]);
	}

	public int getGibbsCounts(int layer)
	{
		return gibbsCounts == null ? 1 : 
			(gibbsCounts.length <= layer ? gibbsCounts[0] : gibbsCounts[layer]);
	}

	public boolean getUseAdaGrads(int layer)
	{
		return useAdaGrads == null ? false : 
			(useAdaGrads.length <= layer ? useAdaGrads[0] : useAdaGrads[layer]);
	}

	public OptimizeType getOptimizeTypes(int layer)
	{
		return optimizeTypes == null ? OptimizeType.LINE_SEARCH : 
			(optimizeTypes.length <= layer ? optimizeTypes[0] : optimizeTypes[layer]);
	}

	public double getMomentums(int layer)
	{
		return momentums == null ? 0 : 
			(momentums.length <= layer ? momentums[0] : momentums[layer]);
	}

	public int getVisibleType()
	{
		return visibleType;
	}

	public int getHiddenType()
	{
		return hiddenType;
	}

	public boolean isFineTuneUseAdaGrads()
	{
		return fineTuneUseAdaGrads;
	}
	
	
}
