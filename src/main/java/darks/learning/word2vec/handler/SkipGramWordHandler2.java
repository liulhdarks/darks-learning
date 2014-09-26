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
package darks.learning.word2vec.handler;

import java.util.List;

import org.jblas.DoubleMatrix;

import darks.learning.common.basic.HaffNode;
import darks.learning.common.basic.WeightHaffNode;
import darks.learning.word2vec.Word2Vec;
import darks.learning.word2vec.WordHandler;
import darks.learning.word2vec.WordNode;

/**
 * Word2vec Skip-gram algorithm
 * 
 * @author Darks.Liu
 *
 */
public class SkipGramWordHandler2 extends WordHandler
{

	public SkipGramWordHandler2(Word2Vec word3vec)
	{
		super(word3vec);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void handle(int index, List<WordNode> sentence)
	{
		long nextRandom = config.randomFunction.randLong();
		int scope = (int) nextRandom % config.window;
		int maxScope = config.window * 2 + 1 - scope;
		int size = sentence.size();
		DoubleMatrix error = new DoubleMatrix(config.featureSize);
		inputToHidden(index, scope, maxScope, size, sentence, error);
	}


	//input -> hidden
	private void inputToHidden(int index, int scope, int maxScope, int size, 
			List<WordNode> sentence, DoubleMatrix error)
	{
		WordNode word = sentence.get(index);
		int c = 0;
		for (int i = scope; i < maxScope; i++)
		{
			if (i == config.window)
			{
				continue;
			}
			c = index - config.window + i;
			if (c < 0 || c >= size)
			{
				continue;
			}
			WordNode curWord = sentence.get(c);
			if (curWord == null)
			{
				continue;
			}
			hierarchySoftmax(word, error, curWord.feature);
			curWord.feature.addi(error);
		}
	}
	
	private void hierarchySoftmax(WordNode word, DoubleMatrix error, DoubleMatrix input)
	{
		List<HaffNode> neuronNodes = word.codeNodes;
		int nodeSize = neuronNodes.size();
		int maxExp = config.maxExp;
		int expTableSize = config.expTableSize;
		for (int i = 0; i < nodeSize; i++)
		{
			WeightHaffNode neuronNode = (WeightHaffNode) neuronNodes.get(i);
			double f = input.dot(neuronNode.weight);
			if (f <= -maxExp)
				continue;
			else if (f >= maxExp)
				continue;
			else
				f = expTable[(int) ((f + maxExp) * (expTableSize / maxExp / 2))];
			double g = (1 - word.codePath[i] - f) * learnRate;
			error.addi(neuronNode.weight.mul(g));
			neuronNode.weight.addi(input.mul(g));
		}
	}
}
