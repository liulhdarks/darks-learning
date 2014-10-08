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
	
	int featureSize = 100;
	
	Word2VecType trainType = Word2VecType.SKIP_GRAM;
	
	int window = 5;
	
	double sample = 1e-3;
	
	double learnRate = 0.025;
	
	int negative = 5;
	
	boolean hierarchicalSoftmax = true;
	
	boolean useNegativeSampling = true;
	
	int maxExp = 6;
	
	int expTableSize = 1000;
	
	RandomFunction randomFunction = new WordRandomFunction();
	
	WordHandler wordHandler;
	
	int minVocabCount = 1;
	
	int topCount = 10;
	
	int unigramTableSize = (int)1e8;
	
	public Word2VecConfig()
	{
		
	}

	/**
	 * Set learn feature size each word.Default 100
	 * @param featureSize Feature size
	 * @return this
	 */
	public Word2VecConfig setFeatureSize(int featureSize)
	{
		this.featureSize = featureSize;
		return this;
	}

	/**
	 * Set train type.{@linkplain darks.learning.word2vec.Word2Vec.Word2VecType Word2VecType}.
	 * Default SKIP_GRAM
	 * @param trainType Train type
	 * @return this
	 */
	public Word2VecConfig setTrainType(Word2VecType trainType)
	{
		this.trainType = trainType;
		return this;
	}

	/**
	 * Set context window size for gram.Default 5
	 * @param window Context window size
	 * @return this
	 */
	public Word2VecConfig setWindow(int window)
	{
		this.window = window;
		return this;
	}

	/**
	 * Set learn rate which will be updated when training dynamically.Default 0.025
	 * @param learnRate Learn reate
	 * @return this
	 */
	public Word2VecConfig setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
		return this;
	}

	/**
	 * Set negative sampling size. if it's less than 1, it won't do negative sampling.Default 5.
	 * @param negative Negative sampling size
	 * @return this
	 */
	public Word2VecConfig setNegative(int negative)
	{
		this.negative = negative;
		return this;
	}

	/**
	 * Enable using negative sampling algorithm.Default true
	 * @param useNegativeSampling If true, negative sampling algorithm will be used.
	 * @return this
	 */
	public Word2VecConfig setUseNegativeSampling(boolean useNegativeSampling)
	{
		this.useNegativeSampling = useNegativeSampling;
		return this;
	}

	/**
	 * Set exp maximum value.Default 6
	 * @param maxExp Maximum exp value
	 * @return this
	 */
	public Word2VecConfig setMaxExp(int maxExp)
	{
		this.maxExp = maxExp;
		return this;
	}

	/**
	 * Set exp function table maximum size.Default 1000
	 * @param maxExp Maximum exp table size
	 * @return this
	 */
	public Word2VecConfig setExpTableSize(int expTableSize)
	{
		this.expTableSize = expTableSize;
		return this;
	}

	/**
	 * Random function.Default {@linkplain darks.learning.common.rand.WordRandomFunction WordRandomFunction}
	 * @param randomFunction Random function
	 * @return this
	 */
	public Word2VecConfig setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
		return this;
	}

	/**
	 * Set sub sample rate.Default 1e-3
	 * @param sample Sub sample rate
	 * @return this
	 */
	public Word2VecConfig setSample(double sample)
	{
		this.sample = sample;
		return this;
	}

	/**
	 * Customize word algorithm.Default null.
	 * @param wordHandler Word algorithm handler
	 * @return this
	 */
	public Word2VecConfig setWordHandler(WordHandler wordHandler)
	{
		this.wordHandler = wordHandler;
		return this;
	}

	/**
	 * Set minimum vocabulary frequency value.Default 1
	 * @param minVocabCount Minimum vocabulary frequency
	 * @return thiss
	 */
	public Word2VecConfig setMinVocabCount(int minVocabCount)
	{
		this.minVocabCount = minVocabCount;
		return this;
	}

	/**
	 * Distance result top count
	 * @param topCount Result top count
	 * @return this
	 */
	public Word2VecConfig setTopCount(int topCount)
	{
		this.topCount = topCount;
		return this;
	}

	/**
	 * Set negative sampling vocabulary table size.Default 1e8
	 * @param negVocabSize Vocabulary table size
	 * @return this
	 */
	public Word2VecConfig setUnigramTableSize(int unigramTableSize)
	{
		this.unigramTableSize = unigramTableSize;
		return this;
	}

	/**
	 * Enable using hierarchical softmax.Default true
	 * @param hierarchicalSoftmax Default true
	 * @return this
	 */
	public Word2VecConfig enableHierarchicalSoftmax(boolean hierarchicalSoftmax)
	{
		this.hierarchicalSoftmax = hierarchicalSoftmax;
		return this;
	}
	
	
}
