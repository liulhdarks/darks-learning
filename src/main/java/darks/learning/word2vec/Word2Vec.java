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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.basic.Haffman;
import darks.learning.corpus.Corpus;
import darks.learning.word2vec.handler.CBowWordHandler;
import darks.learning.word2vec.handler.SkipGramWordHandler;


/**
 * Word2vec implement
 * @author Darks.Liu
 *
 */
public class Word2Vec
{
	
	private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
	
	public enum Word2VecType
	{
		CBOW, SKIP_GRAM
	}
	
	public Word2VecConfig config;
	
	private Map<String, WordNode> wordNodes = new HashMap<String, WordNode>();
	
	private long totalVocabCount = 0;
	
	private DoubleMatrix expTable;
	
	private long actualVocabCount = 0;
	
	private long trainingVocabCount = 0;
	
	double startLearnRate;
	
	double learnRate;
	
	WordHandler cbowHandler = new CBowWordHandler();
	
	WordHandler skipGramHandler = new SkipGramWordHandler();
	
	public Word2Vec()
	{
		
	}
	
	/**
	 * Train wrod2vec model
	 * @param corpus Corpus
	 */
	public void train(Corpus corpus)
	{
		createExpTable();
		buildWordNodes(corpus);
		startTrainer(corpus);
	}
	
	private void buildWordNodes(Corpus corpus)
	{
		totalVocabCount = corpus.getWordFreq().getUniqueCount();
		Iterator<Map.Entry<Comparable<?>, Long>> it = corpus.getWordFreq().entrySetIterator();
		while (it.hasNext())
		{
			Map.Entry<Comparable<?>, Long> entry = it.next();
			long freq = entry.getValue();
			WordNode node = new WordNode((String)entry.getKey(), (int)freq, config);
			wordNodes.put(node.value, node);
		}
		new Haffman(wordNodes.values()).build(config.featureSize);
		if (log.isDebugEnabled())
		{
			log.debug("Total vacab count " + totalVocabCount + ".Words map size:" + wordNodes.size());
		}
	}
	
	private void createExpTable()
	{
		if (expTable != null)
		{
			return;
		}
		expTable = new DoubleMatrix(config.expTableSize);
		for (int i = 0; i < config.expTableSize; i++)
		{
			double exp = FastMath.exp(i / (config.expTableSize * 2 - 1) * config.maxExp);
			expTable.put(i, exp / (1 + exp));
		}
		if (log.isDebugEnabled())
		{
			log.debug("Create exp table size " + expTable.length);
		}
	}
	
	private void startTrainer(Corpus corpus)
	{
		try
		{
			learnRate = config.learnRate;
			startLearnRate = learnRate;
			long lastTrainCount = 0;
			String line = null;
			long nextRandom = 5;
			while ((line = corpus.readCorpusLine()) != null)
			{
				line = line.trim();
				if ("".equals(line))
				{
					continue;
				}
				StringTokenizer token = new StringTokenizer(line, " \t");
				trainingVocabCount = token.countTokens();
				lastTrainCount = updateLearnRate(lastTrainCount);
				List<WordNode> sentence = sampleSentence(nextRandom, token);
				if (sentence.isEmpty())
				{
					continue;
				}
				executeNeuronNetwork(nextRandom, sentence);
			}
		}
		catch (Exception e)
		{
			log.error(e.getMessage(), e);
		}
		finally
		{
			corpus.closeReader();
		}
	}
	
	private long updateLearnRate(long lastTrainCount)
	{
		if (actualVocabCount - lastTrainCount > 10000)
		{
			actualVocabCount += trainingVocabCount;
			learnRate = startLearnRate * (1 - actualVocabCount / (totalVocabCount + 1));
			if (learnRate > startLearnRate * 1e-3)
			{
				learnRate = startLearnRate * 1e-3;
			}
			lastTrainCount = actualVocabCount;
		}
		return lastTrainCount;
	}
	
	private List<WordNode> sampleSentence(long nextRandom, StringTokenizer token)
	{
		List<WordNode> sentence = new ArrayList<WordNode>();
		double sample = config.sample;
		while (token.hasMoreTokens())
		{
			String word = token.nextToken();
			WordNode node = wordNodes.get(word);
			if (node == null)
			{
				continue;
			}
			if (config.sample > 0)
			{
				double rnd = (1 + FastMath.sqrt(node.freq / (sample * totalVocabCount))) 
						* (sample * totalVocabCount) / node.freq;
				nextRandom = nextRandom * 25214903917L + 11;
				if (rnd < (nextRandom & 0xFFFF) / (double) 65536)
				{
					continue;
				}
			}
			sentence.add(node);
		}
		return sentence;
	}
	
	private void executeNeuronNetwork(long nextRandom, List<WordNode> sentence)
	{
		int size = sentence.size();
		for (int i = 0; i < size; i++)
		{
			nextRandom = nextRandom * 25214903917L + 11;
			if (config.trainType == Word2VecType.CBOW)
			{
				cbowHandler.handle(i, sentence, (int) nextRandom % config.window);
			}
			else
			{
				cbowHandler.handle(i, sentence, (int) nextRandom % config.window);
			}
		}
	}
	
	public Word2VecConfig getConfig()
	{
		return config;
	}


	
}
