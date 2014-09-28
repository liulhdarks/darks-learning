/**
 * 
 * Copyright 2014 The Darks Learning Project (Liu lihua)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */
package darks.learning.word2vec;

import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;

public abstract class WordHandler
{

	protected Word2VecConfig config;

	protected double[] expTable;

	protected Map<String, WordNode> wordNodeMap;

	protected WordNode[] unigramTable;

	protected double learnRate;

	public WordHandler()
	{
	}

	public WordHandler(Word2Vec word3vec)
	{
		config = word3vec.getConfig();
		expTable = word3vec.getExpTable();
		learnRate = word3vec.getLearnRate();
		wordNodeMap = word3vec.getWordNodes();
		unigramTable = word3vec.getUnigramTable();
	}

	/**
	 * Handle word2vec algorithm
	 * 
	 * @param index The word index of sentence
	 * @param sentence words set in sentence
	 * @param winScope Random window scope
	 */
	public abstract void handle(int index, List<WordNode> sentence);

	protected void negaticeSampling(WordNode word, DoubleMatrix error, List<WordNode> sentence)
	{
		if (config.negative <= 0)
		{
			return;
		}
		int tableSize = config.unigramTableSize;
		int label = 0;
		WordNode target = null;
		for (int i = 0; i < config.negative + 1; i++)
		{
			if (i == 0)
			{
				target = word;
				label = 1;
			}
			else
			{
				int index = config.randomFunction.randInt(tableSize);
				target = unigramTable[index];
				label = 0;
				if (target.name.equals(word.name))
				{
					continue;
				}
//				long rand = config.randomFunction.randLong();
//				int index = (int)((rand >> 16) % tableSize);
//				if (index >= 0 && index < unigramTable.length)
//				{
//					target = unigramTable[index];
//				}
//				else
//				{
//					rand = Math.abs(rand);
//					index = (int)(rand % sentence.size());
//					target = sentence.get(index);
//				}
			}
			if (target == null)
			{
				return;
			}
			int maxExp = config.maxExp;
			int expTableSize = config.expTableSize;
			double f = word.feature.dot(target.weightNeg);
			double g = 0;
			if (f > maxExp)
				g = (label - 1) * learnRate;
			else if (f < -maxExp)
				g = (label - 0) * learnRate;
			else
				g = (label - expTable[(int) ((f + maxExp) * (expTableSize / maxExp / 2))])
						* learnRate;
			error.addi(target.weightNeg.mul(g));
			target.weightNeg.addi(word.feature.mul(g));
		}
	}

	public double getLearnRate()
	{
		return learnRate;
	}

	public void setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
	}

}
