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
package darks.learning.word2vec;

import darks.learning.common.rand.RandomFunction;
import darks.learning.common.rand.WordRandomFunction;
import darks.learning.word2vec.Word2Vec.Word2VecType;

/**
 * Build word2vec train parameters.
 * @author Darks.Liu
 *
 */
public class Word2VecConfig
{
	
	public int featureSize = 100;
	
	public Word2VecType trainType = Word2VecType.SKIP_GRAM;
	
	public int window = 5;
	
	public double sample = 1e-3;
	
	public double learnRate = 0.025;
	
	public int negative = 5;
	
	public boolean useNegativeSampling = true;
	
	public int negVocabSize = (int) 1e8;
	
	public int maxExp = 6;
	
	public int expTableSize = 1000;
	
	public RandomFunction randomFunction = new WordRandomFunction();
	
	public boolean useStopwords = true;
	
	public WordHandler wordHandler;
	
	public int minVocabCount = 5;
	
	public int topCount = 10;
	
	public Word2VecConfig()
	{
		
	}

	public Word2VecConfig setFeatureSize(int featureSize)
	{
		this.featureSize = featureSize;
		return this;
	}

	public Word2VecConfig setTrainType(Word2VecType trainType)
	{
		this.trainType = trainType;
		return this;
	}

	public Word2VecConfig setWindow(int window)
	{
		this.window = window;
		return this;
	}

	public Word2VecConfig setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
		return this;
	}

	public Word2VecConfig setNegative(int negative)
	{
		this.negative = negative;
		return this;
	}

	public Word2VecConfig setUseNegativeSampling(boolean useNegativeSampling)
	{
		this.useNegativeSampling = useNegativeSampling;
		return this;
	}

	public Word2VecConfig setNegVocabSize(int negVocabSize)
	{
		this.negVocabSize = negVocabSize;
		return this;
	}

	public Word2VecConfig setMaxExp(int maxExp)
	{
		this.maxExp = maxExp;
		return this;
	}

	public Word2VecConfig setExpTableSize(int expTableSize)
	{
		this.expTableSize = expTableSize;
		return this;
	}

	public Word2VecConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	public Word2VecConfig setUseStopwords(boolean useStopwords)
	{
		this.useStopwords = useStopwords;
		return this;
	}

	public Word2VecConfig setSample(double sample)
	{
		this.sample = sample;
		return this;
	}

	public Word2VecConfig setWordHandler(WordHandler wordHandler)
	{
		this.wordHandler = wordHandler;
		return this;
	}

	public Word2VecConfig setMinVocabCount(int minVocabCount)
	{
		this.minVocabCount = minVocabCount;
		return this;
	}

	public Word2VecConfig setTopCount(int topCount)
	{
		this.topCount = topCount;
		return this;
	}
	
	
}
