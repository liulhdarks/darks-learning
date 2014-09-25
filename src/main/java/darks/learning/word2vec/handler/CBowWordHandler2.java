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
 * Word2vec CBOW algorithm
 * 
 * @author Darks.Liu
 *
 */
public class CBowWordHandler2 extends WordHandler
{
	
	
	public CBowWordHandler2(Word2Vec word3vec)
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
		int b = (int) nextRandom % config.window;
		cbowGram(index, sentence, b);
	}
	
	private void cbowGram(int index, List<WordNode> sentence, int b)
	{
		WordNode word = sentence.get(index);
		int maxExp = config.maxExp;
		int expTableSize = config.expTableSize;
		int a, c = 0;

		List<HaffNode> neurons = word.codeNodes;
		double[] neu1e = new double[config.featureSize];// 误差项
		double[] neu1 = new double[config.featureSize];// 误差项
		WordNode last_word;
		int window = config.window;
		int layerSize = config.featureSize;
		for (a = b; a < window * 2 + 1 - b; a++)
			if (a != window)
			{
				c = index - window + a;
				if (c < 0)
					continue;
				if (c >= sentence.size())
					continue;
				last_word = sentence.get(c);
				if (last_word == null)
					continue;
				for (c = 0; c < layerSize; c++)
					neu1[c] += last_word.feature.data[c];
			}

		// HIERARCHICAL SOFTMAX
		for (int d = 0; d < neurons.size(); d++)
		{
			WeightHaffNode out = (WeightHaffNode) neurons.get(d);
			// System.out.println(Arrays.toString(out.syn1));
			double f = 0;
			// Propagate hidden -> output
			for (c = 0; c < layerSize; c++)
				f += neu1[c] * out.weight.data[c];
			if (f <= -maxExp)
				continue;
			else if (f >= maxExp)
				continue;
			else
				f = expTable[(int) ((f + maxExp) * (expTableSize / maxExp / 2))];
			// 'g' is the gradient multiplied by the learning rate
			double g = (1 - word.codePath[d] - f) * learnRate;
			// double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
			// double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;
			//
			for (c = 0; c < layerSize; c++)
			{
				neu1e[c] += g * out.weight.data[c];
			}
			// Learn weights hidden -> output
			for (c = 0; c < layerSize; c++)
			{
				out.weight.data[c] += g * neu1[c];
			}
		}
		for (a = b; a < window * 2 + 1 - b; a++)
		{
			if (a != window)
			{
				c = index - window + a;
				if (c < 0)
					continue;
				if (c >= sentence.size())
					continue;
				last_word = sentence.get(c);
				if (last_word == null)
					continue;
				for (c = 0; c < layerSize; c++)
					last_word.feature.data[c] += neu1e[c];
			}

		}
	}

}
