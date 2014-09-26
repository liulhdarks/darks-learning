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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.basic.Haffman;
import darks.learning.common.utils.IOUtils;
import darks.learning.corpus.Corpus;
import darks.learning.word2vec.Word2Vec.Word2VecType;
import darks.learning.word2vec.handler.CBowWordHandler2;

/**
 * Word2vec implement
 * 
 * @author Darks.Liu
 * 
 */
public class Word2Vec2
{

	private static Logger log = LoggerFactory.getLogger(Word2Vec2.class);

	public Word2VecConfig config = new Word2VecConfig();

	private Map<String, WordNode> wordNodes = new HashMap<String, WordNode>();

	private long totalVocabCount = 0;

	private double[] expTable;

	private long actualVocabCount = 0;

	private long trainingVocabCount = 0;

	double startLearnRate;

	double learnRate;

	WordHandler wordHandler = null;
	
	int outCount = 0;

	public Word2Vec2()
	{

	}

	/**
	 * Train wrod2vec model
	 * 
	 * @param corpus Corpus
	 */
	public void train(Corpus corpus)
	{
		log.info("Start to train word2vec corpus.");
		long st = System.currentTimeMillis();
		createExpTable();
		buildWordNodes(corpus);
		initTrainer();
		startTrainer(corpus);
		log.info("Complete to train word2vec corpus.Used time:" + (System.currentTimeMillis() - st)
				/ 1000 + "s");
	}

	private void buildWordNodes(Corpus corpus)
	{
		totalVocabCount = corpus.getTotalVocabCount();
		Iterator<Map.Entry<Comparable<?>, Long>> it = corpus.getWordFreq().entrySetIterator();
		while (it.hasNext())
		{
			Map.Entry<Comparable<?>, Long> entry = it.next();
			long freq = entry.getValue();
			if (freq < config.minVocabCount)
			{
				continue;
			}
			WordNode node = new WordNode((String) entry.getKey(), (int) freq, config);
			wordNodes.put(node.name, node);
		}
		if (log.isDebugEnabled())
		{
			log.debug("Total vacab count " + totalVocabCount + ".Words map size:"
					+ wordNodes.size());
			log.debug("Building haffman for word2vec node.");
		}
		new Haffman(wordNodes.values()).build(config.featureSize);
		if (log.isDebugEnabled())
		{
			log.debug("Complete to Build haffman tree.");
		}
		PrintWriter out = null;
		try
		{
			File file2 = new File("test/tmp.txt");
			out = new PrintWriter(file2);
		}
		catch (FileNotFoundException e)
		{
		}
		for (WordNode neuron : wordNodes.values())
		{
			WordNode node = (WordNode) neuron;
			out.print(node.name + "\t" + node.value + "\t" + node.code + "\t" + Arrays.toString(node.codePath) + "\r\n");
		}
		out.flush();
		out.close();
	}

	private void createExpTable()
	{
		if (expTable != null)
		{
			return;
		}
		expTable = new double[config.expTableSize];
		for (int i = 0; i < config.expTableSize; i++)
		{
			double exp = FastMath.exp(i / (config.expTableSize * 2 - 1) * config.maxExp);
			expTable[i] = exp / (1 + exp);
		}
		if (log.isDebugEnabled())
		{
			log.debug("Create exp table size " + expTable.length);
		}
	}

	private void initTrainer()
	{
		learnRate = config.learnRate;
		startLearnRate = learnRate;
		if (config.wordHandler != null)
		{
			wordHandler = config.wordHandler;
		}
		else
		{
			if (config.trainType == Word2VecType.CBOW)
			{
				wordHandler = new CBowWordHandler2(this);
			}
			else
			{
				//wordHandler = new SkipGramWordHandler2(this);
			}
		}
	}

	private void startTrainer(Corpus corpus)
	{
		try
		{
			log.debug("Start to train with " + wordHandler);
			long lastTrainCount = 0;
			String line = null;
			while ((line = corpus.readCorpusLine()) != null)
			{
				line = line.trim();
				if ("".equals(line))
				{
					continue;
				}
				StringTokenizer token = new StringTokenizer(line, " \t");
				trainingVocabCount = token.countTokens();
				List<WordNode> sentence = sampleSentence(token);
				if (sentence.isEmpty())
				{
					continue;
				}
				lastTrainCount = updateLearnRate(lastTrainCount);
				executeNeuronNetwork(sentence);
			}
			System.out.println("Words in train file: " + actualVocabCount + "/" + totalVocabCount);
			System.out.println("sucess train over!");
			System.out.println("out range " + outCount);
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
		actualVocabCount += trainingVocabCount;
		if (actualVocabCount - lastTrainCount > 10000)
		{
			log.debug("learnRate:" + learnRate + "\tProgress: "
					+ (int) (actualVocabCount / (double) (totalVocabCount + 1) * 100) + "%");
			learnRate = startLearnRate * (1. - actualVocabCount / (double) (totalVocabCount + 1));
			if (learnRate < startLearnRate * 1e-3)
			{
				learnRate = startLearnRate * 1e-3;
			}
			lastTrainCount = actualVocabCount;
		}
		wordHandler.setLearnRate(learnRate);
		return lastTrainCount;
	}

	private List<WordNode> sampleSentence(StringTokenizer token)
	{
		List<WordNode> sentence = new ArrayList<WordNode>();
		double sample = config.sample;
		while (token.hasMoreTokens())
		{
			String word = token.nextToken();
			WordNode node = wordNodes.get(word);
			if (node == null)
			{
				trainingVocabCount--;
				continue;
			}
			if (config.sample > 0)
			{
				double rnd = (1 + FastMath.sqrt(node.value / (sample * totalVocabCount)))
						* (sample * totalVocabCount) / node.value;
				double nextRandom = config.randomFunction.randDouble();
				if (rnd < nextRandom)
				{
					outCount++;
					continue;
				}
			}
			sentence.add(node);
		}
		return sentence;
	}

	private void executeNeuronNetwork(List<WordNode> sentence)
	{
		int size = sentence.size();
		for (int i = 0; i < size; i++)
		{
			wordHandler.handle(i, sentence);
		}
	}
	
	/**
	 * Calculate similar between two specify words
	 * 
	 * @param word1 Target word one
	 * @param word2 Target word two
	 * @return Similar score
	 */
	public double distance(String word1, String word2)
	{
		WordNode node1 = wordNodes.get(word1);
		WordNode node2 = wordNodes.get(word2);
		if (node1 == null || node2 == null)
		{
			return 0.001;
		}
		return node1.feature.dot(node2.feature);
	}

	/**
	 * Calculate specify word's nearest or relate words
	 * 
	 * @param word Specify word
	 * @return Nearest or relate words
	 */
	public Set<WordEntry> distance(String word)
	{
		int resultSize = FastMath.min(config.topCount, wordNodes.size());
		TreeSet<WordEntry> result = new TreeSet<WordEntry>();
		WordNode node = wordNodes.get(word);
		if (node != null)
		{
			double minSim = Double.MIN_VALUE;
			for (WordNode target : wordNodes.values())
			{
				if (target.name.equals(word))
				{
					continue;
				}
				double sim = target.feature.dot(node.feature);
				if (sim > minSim)
				{
					result.add(new WordEntry(target.name, sim));
					if (result.size() > resultSize)
					{
						result.pollLast();
					}
					minSim = result.last().similar;
				}
			}
		}
		return result;
	}
	
	public double[] getWordVector(String word) {
		WordNode node = wordNodes.get(word);
		if (node == null)
		{
			return null;
		}
		return node.feature.toArray();
	}


	/**
	 * Save model
	 * 
	 * @param file Model file saved
	 */
	public void saveModel(File file)
	{
		log.info("Saving word2vec model to " + file);
		DataOutputStream dos = null;
		try
		{
			dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
			dos.writeInt(wordNodes.size());
			dos.writeInt(config.featureSize);
			double[] syn0 = null;
			for (Entry<String, WordNode> element : wordNodes.entrySet())
			{
				byte[] bytes = element.getKey().getBytes();
				dos.writeInt(bytes.length);
				dos.write(bytes);
				syn0 = (element.getValue()).feature.toArray();
				for (int i = 0; i < config.featureSize; i++)
				{
					dos.writeDouble(syn0[i]);
				}
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
		finally
		{
			IOUtils.closeStream(dos);
		}
	}

	/**
	 * Load model file
	 * 
	 * @param file Model file
	 */
	public void loadModel(File file)
	{
		DataInputStream dis = null;
		try
		{
			log.info("Reading word2vec model from " + file);
			dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
			int wordSize = dis.readInt();
			int featureSize = dis.readInt();
			wordNodes.clear();
			for (int i = 0; i < wordSize; i++)
			{
				DoubleMatrix feature = new DoubleMatrix(featureSize);
				int nameLen = dis.readInt();
				byte[] bits = new byte[nameLen];
				dis.read(bits, 0, nameLen);
				String name = new String(bits);
				double len = 0;
				for (int f = 0; f < featureSize; f++)
				{
					double w = dis.readDouble();
					feature.put(f, w);
					len += w * w;
				}
				len = FastMath.sqrt(len);
				feature.divi(len);
				wordNodes.put(name, new WordNode(name, feature));
			}
			log.info("Succeed to read word2vec model. Word dictionary size " + wordNodes.size());
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		finally
		{
			IOUtils.closeStream(dis);
		}
	}
	
	public void saveModel2(File file)
	{
		try 
		{
			DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
					new FileOutputStream(file)));
			dataOutputStream.writeInt(wordNodes.size());
			dataOutputStream.writeInt(config.featureSize);
			double[] syn0 = null;
			for (Entry<String, WordNode> element : wordNodes.entrySet())
			{
				dataOutputStream.writeUTF(element.getKey());
				syn0 = ((WordNode) element.getValue()).feature.data;
				for (double d : syn0)
				{
					dataOutputStream.writeFloat(((Double) d).floatValue());
				}
			}
			dataOutputStream.flush();
			dataOutputStream.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
	
	/**
	 * 
	 * @author Darks.Liu
	 * 
	 */
	public static class WordEntry implements Comparable<WordEntry>
	{
		public String name;

		public double similar;
		

		public WordEntry(String name, double similar)
		{
			super();
			this.name = name;
			this.similar = similar;
		}

		@Override
		public String toString()
		{
			return this.name + "\t" + similar;
		}

		@Override
		public int compareTo(WordEntry o)
		{
			if (this.similar < o.similar)
			{
				return 1;
			}
			else
			{
				return -1;
			}
		}

	}

	public Word2VecConfig getConfig()
	{
		return config;
	}

	public double[] getExpTable()
	{
		return expTable;
	}

	public double getLearnRate()
	{
		return learnRate;
	}

	public Map<String, WordNode> getWordNodes()
	{
		return wordNodes;
	}

}