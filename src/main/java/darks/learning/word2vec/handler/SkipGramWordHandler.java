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
public class SkipGramWordHandler extends WordHandler
{

	public int EXP_TABLE_SIZE = 1000;

	private int MAX_EXP = 6;

	public SkipGramWordHandler(Word2Vec word3vec)
	{
		super(word3vec);
	}
	
	private void createExpTable()
	{
		for (int i = 0; i < EXP_TABLE_SIZE; i++)
		{
			expTable[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
			expTable[i] = expTable[i] / (expTable[i] + 1);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void handle(int index, List<WordNode> sentence)
	{
		int maxExp = config.maxExp;
		int expTableSize = config.expTableSize;
		long nextRandom = config.randomFunction.randLong();
		int b = (int) nextRandom % config.window;
		WordNode word = sentence.get(index);
		int a, c = 0;
		int window = config.window;
		int layerSize = config.featureSize;
		for (a = b; a < window * 2 + 1 - b; a++)
		{
			if (a == window)
			{
				continue;
			}
			c = index - window + a;
			if (c < 0 || c >= sentence.size())
			{
				continue;
			}

			double[] neu1e = new double[layerSize];// 误差项
			// HIERARCHICAL SOFTMAX
			List<HaffNode> neurons = word.codeNodes;
			WordNode we = sentence.get(c);
			for (int i = 0; i < neurons.size(); i++)
			{
				WeightHaffNode out = (WeightHaffNode) neurons.get(i);
				double f = 0;
				// Propagate hidden -> output
				for (int j = 0; j < layerSize; j++)
				{
					f += we.feature.data[j] * out.weight.data[j];
				}
				if (f <= -MAX_EXP || f >= MAX_EXP)
				{
					continue;
				}
				else
				{
					f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
					f = expTable[(int) f];
				}
				// 'g' is the gradient multiplied by the learning rate
				double g = (1 - word.codePath[i] - f) * learnRate;
				// Propagate errors output -> hidden
				for (c = 0; c < layerSize; c++)
				{
					neu1e[c] += g * out.weight.data[c];
				}
				// Learn weights hidden -> output
				for (c = 0; c < layerSize; c++)
				{
					out.weight.data[c] += g * we.feature.data[c];
				}
			}

			// Learn weights input -> hidden
			for (int j = 0; j < layerSize; j++)
			{
				we.feature.data[j] += neu1e[j];
			}
		}
	}
}
