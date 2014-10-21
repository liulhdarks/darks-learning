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

import darks.learning.LearningConfig;
import darks.learning.SupervisedLearning;
import darks.learning.common.rand.RandomFunction;
import darks.learning.exceptions.ConfigException;
import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.activate.ActivateFunction;
import darks.learning.neuron.activate.Activations;
import darks.learning.optimize.LearningOptimizer.OptimizeType;

/**
 * DBN Configuration
 * 
 * @author Darks.Liu
 *
 */
public class SdaConfig extends LearningConfig
{

	public enum FinetuneType
	{
		SOFTMAX, LOGISTIC, BP
	}
	
	double fineTuneLearnRate = 1e-3;
	
	int fineTuneIterateCount = 10000;
	
	ActivateFunction fineTuneActivation;
	
	int fineTuneLossType = LossFunction.LOGLIKELIHOOD_LOSS; 
	
	FinetuneType finetuneType = FinetuneType.SOFTMAX;
	
	boolean fineTuneUseAdaGrads = false;
	
	SupervisedLearning fineTuneLayer;
	
	double[] hiddensLearnRate;
	
	int[] hiddensIterateCount;
	
	int[] hiddenLayouts;
	
	int[] hiddenLossTypes;
	
	boolean[] useAdaGrads;
	
	OptimizeType[] optimizeTypes;
	
	double[] momentums;
	
	boolean useSample = false;
	
	double corruptionLevel = 0.3;
	
	public SdaConfig()
	{
		
	}
	
	public void checkConfigValid()
	{
		if (hiddenLayouts == null)
		{
			throw new ConfigException("DBN's hidden layouts cannot be null");
		}
	}

	public SdaConfig setFineTuneLearnRate(double fineTuneLearnRate)
	{
		this.fineTuneLearnRate = fineTuneLearnRate;
		return this;
	}

	public SdaConfig setFineTuneIterateCount(int fineTuneIterateCount)
	{
		this.fineTuneIterateCount = fineTuneIterateCount;
		return this;
	}

	public SdaConfig setFineTuneActivation(ActivateFunction fineTuneActivation)
	{
		this.fineTuneActivation = fineTuneActivation;
		return this;
	}

	public SdaConfig setFineTuneLossType(int fineTuneLossType)
	{
		this.fineTuneLossType = fineTuneLossType;
		return this;
	}

	public SdaConfig setHiddensLearnRate(double hiddensLearnRate)
	{
		this.hiddensLearnRate = new double[]{hiddensLearnRate};
		return this;
	}

	public SdaConfig setHiddensLearnRate(double[] hiddensLearnRate)
	{
		this.hiddensLearnRate = hiddensLearnRate;
		return this;
	}

	public SdaConfig setHiddensIterateCount(int hiddensIterateCount)
	{
		this.hiddensIterateCount = new int[]{hiddensIterateCount};
		return this;
	}

	public SdaConfig setHiddensIterateCount(int[] hiddensIterateCount)
	{
		this.hiddensIterateCount = hiddensIterateCount;
		return this;
	}

	public SdaConfig setHiddenLayouts(int hiddenLayouts)
	{
		this.hiddenLayouts = new int[]{hiddenLayouts};
		return this;
	}

	public SdaConfig setHiddenLayouts(int[] hiddenLayouts)
	{
		this.hiddenLayouts = hiddenLayouts;
		return this;
	}

	public SdaConfig setHiddenLossTypes(int hiddenLossTypes)
	{
		this.hiddenLossTypes = new int[]{hiddenLossTypes};
		return this;
	}

	public SdaConfig setHiddenLossTypes(int[] hiddenLossTypes)
	{
		this.hiddenLossTypes = hiddenLossTypes;
		return this;
	}
	
	public SdaConfig setUseAdaGrads(boolean useAdaGrads)
	{
		this.useAdaGrads = new boolean[]{useAdaGrads};
		return this;
	}

	public SdaConfig setUseAdaGrads(boolean[] useAdaGrads)
	{
		this.useAdaGrads = useAdaGrads;
		return this;
	}

	public SdaConfig setOptimizeTypes(OptimizeType optimizeTypes)
	{
		this.optimizeTypes = new OptimizeType[]{optimizeTypes};
		return this;
	}

	public SdaConfig setOptimizeTypes(OptimizeType[] optimizeTypes)
	{
		this.optimizeTypes = optimizeTypes;
		return this;
	}

	public SdaConfig setMomentums(double momentums)
	{
		this.momentums = new double[]{momentums};
		return this;
	}

	public SdaConfig setMomentums(double[] momentums)
	{
		this.momentums = momentums;
		return this;
	}


	public SdaConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public SdaConfig setNormalized(boolean normalized)
	{
		this.normalized = normalized;
		return this;
	}

	public SdaConfig setUseSample(boolean useSample)
	{
		this.useSample = useSample;
		return this;
	}
	
	public SdaConfig setFinetuneType(FinetuneType finetuneType)
	{
		this.finetuneType = finetuneType;
		return this;
	}

	public SdaConfig setFineTuneLayer(SupervisedLearning fineTuneLayer)
	{
		this.fineTuneLayer = fineTuneLayer;
		return this;
	}

	public SdaConfig setFineTuneUseAdaGrads(boolean fineTuneUseAdaGrads)
	{
		this.fineTuneUseAdaGrads = fineTuneUseAdaGrads;
		return this;
	}

	public SdaConfig setCorruptionLevel(double corruptionLevel)
	{
		this.corruptionLevel = corruptionLevel;
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

	public int getHiddenLossTypes(int layer)
	{
		return hiddenLossTypes == null ? LossFunction.RECONSTRUCTION_CROSSENTROPY : 
				(hiddenLossTypes.length <= layer ? hiddenLossTypes[0] : hiddenLossTypes[layer]);
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

	public boolean isFineTuneUseAdaGrads()
	{
		return fineTuneUseAdaGrads;
	}

	public double getCorruptionLevel()
	{
		return corruptionLevel;
	}
	
}
