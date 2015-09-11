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
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.ProgressReporter;
import darks.learning.common.basic.Haffman;
import darks.learning.common.utils.IOUtils;
import darks.learning.corpus.Corpus;

/**
 * Word2vec implement
 * 
 * @author Darks.Liu
 * 
 */
public class Word2Vec
{

	public static final String REPORT_PERCENT = "percent";

	public static final String REPORT_LEARN_RATE = "learnRate";
	
	private static Logger log = LoggerFactory.getLogger(Word2Vec.class);

	/**
	 * Word2vec algorithm type.Such as CBOW or Skip-gram
	 * @author Darks.Liu
	 *
	 */
	public enum Word2VecType
	{
		CBOW, SKIP_GRAM
	}
	
	public enum DistanceType
	{
		COSINE, STATISTIC
	}

	public Word2VecConfig config = new Word2VecConfig();

	private Map<String, WordNode> wordNodes = new HashMap<String, WordNode>();
	
	private WordNode[] unigramTable = null;

	private long totalVocabCount = 0;

	private double[] expTable;

	private long actualVocabCount = 0;

	private long trainingVocabCount = 0;
	
	private volatile boolean parallelLoad = false;

	double startLearnRate;

	double learnRate;

	WordHandler wordHandler = null;
	
	ProgressReporter progressReporter;
	
	public Word2Vec()
	{

	}

	/**
	 * Train wrod2vec model
	 * 
	 * @param corpus Corpus
	 */
	public boolean train(Corpus corpus)
	{
		log.info("Start to train word2vec corpus.");
		long st = System.currentTimeMillis();
		if (!buildWordNodes(corpus))
		{
			return false;
		}
		initTrainer();
		startTrainer(corpus);
		log.info("Complete to train word2vec corpus.Used time:" + (System.currentTimeMillis() - st)
				/ 1000 + "s");
		return true;
	}
	
	/**
	 * Incremental training sentence. 
	 * 
	 * @param source Target source
	 * @return Whether success
	 */
	public boolean trainIncrement(String source)
	{
		log.debug("Training word2vec " + source + " by increment way.");
		StringTokenizer token = new StringTokenizer(source, " \t");
		trainingVocabCount = token.countTokens();
		List<WordNode> sentence = sampleSentence(token);
		if (sentence.isEmpty())
		{
			return false;
		}
		executeNeuronNetwork(sentence);
		return true;
	}

	private boolean buildWordNodes(Corpus corpus)
	{
		if (corpus == null || corpus.getTotalVocabCount() == 0)
			return false;
		totalVocabCount = corpus.getTotalVocabCount();
		Iterator<Map.Entry<String, Long>> it = corpus.getWordFreq().entrySetIterator();
		while (it.hasNext())
		{
			Map.Entry<String, Long> entry = it.next();
			int freq = entry.getValue().intValue();
			if (freq < config.minVocabCount)
			{
				continue;
			}
			WordNode node = new WordNode(entry.getKey(), freq, config);
			wordNodes.put(node.name, node);
		}
		if (log.isDebugEnabled())
		{
			log.debug("Total vacab count " + totalVocabCount + ".Words map size:"
					+ wordNodes.size());
			log.debug("Building haffman for word2vec node.");
		}
		if (wordNodes.isEmpty())
		{
			log.error("Training vocab count is zero.");
			return false;
		}
		new Haffman(wordNodes.values()).build(config.featureSize);
		if (log.isDebugEnabled())
		{
			log.debug("Complete to Build haffman tree.");
		}
		return true;
	}

	private void createExpTable()
	{
		if (expTable != null)
		{
			return;
		}
		if (log.isDebugEnabled())
		{
			log.debug("Create exp table size " + config.expTableSize);
		}
		expTable = new double[config.expTableSize + 1];
		for (int i = 0; i <= config.expTableSize; i++)
		{
			double exp = FastMath.exp((i / (double)config.expTableSize * 2 - 1) * config.maxExp);
			expTable[i] = exp / (1 + exp);
		}
	}
	
	private void createUnigramTable()
	{
		if (config.negative <= 0 || config.unigramTableSize <= 0)
		{
			return;
		}
		if (log.isDebugEnabled())
		{
			log.debug("Create unigram table size " + config.unigramTableSize);
		}
		int tableSize = config.unigramTableSize;
		unigramTable = new WordNode[tableSize];
		WordNode[] vocabs = new WordNode[wordNodes.size()];
		wordNodes.values().toArray(vocabs);
		int vocabSize = vocabs.length;
		double pow = 0.75;
		double totalPower = 0;
		for (WordNode vocab : vocabs)
		{
			totalPower += FastMath.pow(vocab.value, pow);
		}
		double dl = FastMath.pow(vocabs[0].value, pow) / totalPower;
		int index = 0;
		for (int i = 0; i < tableSize; i++)
		{
			unigramTable[i] = vocabs[index];
			if (i / (double)tableSize > dl)
			{
				index++;
				dl += FastMath.pow(vocabs[index].value, pow) / totalPower;
			}
			if (index >= vocabSize)
			{
				index = vocabSize - 1;
			}
		}
	}

	private void initTrainer()
	{
		createExpTable();
		createUnigramTable();
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
				wordHandler = new CBowWordHandler(this);
			}
			else
			{
				wordHandler = new SkipGramWordHandler(this);
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
			log.debug("Words in train file: " + actualVocabCount + "/" + totalVocabCount);
			log.info("Succeed to train word2vec model.");
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
			if (learnRate < startLearnRate * 1e-4)
			{
				learnRate = startLearnRate * 1e-4;
			}
			lastTrainCount = actualVocabCount;
			if (progressReporter != null)
			{
				Map<String, Object> params = new HashMap<String, Object>();
				params.put(REPORT_PERCENT, (double)actualVocabCount / (double) (totalVocabCount + 1));
				params.put(REPORT_LEARN_RATE, learnRate);
				progressReporter.progress(params);
			}
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
	 * Calculate similar between two words lists
	 * 
	 * @param sources Source words list
	 * @param targets Target words list
	 * @return Similar score
	 */
	public double distance(Collection<String> sources, Collection<String> targets)
	{
		return distance(sources, null, targets, null);
	}
	
	/**
	 * Calculate similar between two words lists
	 * 
	 * @param sources Source words list
	 * @param targets Target words list
	 * @param distanceType Distance type
	 * @return Similar score
	 */
	public double distance(Collection<String> sources, Collection<String> targets, DistanceType distanceType)
	{
		return distance(sources, null, targets, null, distanceType);
	}
	
	/**
	 * Calculate similar between two words lists with weights
	 * 
	 * @param sources Source words list
	 * @param sourceWeights Source words weight
	 * @param targets Target words list
	 * @param targetWeights Target words weight
	 * @return Similar score
	 */
	public double distance(Collection<String> sources, Map<String, Double> sourceWeights, 
			Collection<String> targets, Map<String, Double> targetWeights)
	{
		return distance(sources, sourceWeights, targets, targetWeights, DistanceType.COSINE);
	}
	
	/**
	 * Calculate similar between two words lists with weights
	 * 
	 * @param sources Source words list
	 * @param sourceWeights Source words weight
	 * @param targets Target words list
	 * @param targetWeights Target words weight
	 * @param distanceType Distance type
	 * @return Similar score
	 */
	public double distance(Collection<String> sources, Map<String, Double> sourceWeights, 
			Collection<String> targets, Map<String, Double> targetWeights, DistanceType distanceType)
	{
		if (distanceType == DistanceType.COSINE)
		{
			return distanceCosine(sources, sourceWeights, targets, targetWeights);
		}
		else
		{
			return distanceStatistic(sources, sourceWeights, targets, targetWeights);
		}
	}
	

	/**
	 * Calculate similar between two words lists with weights by cosine
	 * 
	 * @param sources Source words list
	 * @param sourceWeights Source words weight
	 * @param targets Target words list
	 * @param targetWeights Target words weight
	 * @return Similar score
	 */
	private double distanceCosine(Collection<String> sources, Map<String, Double> sourceWeights, 
			Collection<String> targets, Map<String, Double> targetWeights)
	{
		DoubleMatrix srcMt = getSentenceFeature(sources, sourceWeights);
		DoubleMatrix targetMt = getSentenceFeature(targets, targetWeights);
		if (srcMt == null || targetMt == null)
		{
			return 0.001;
		}
		double sim = srcMt.dot(targetMt);
		sim = sim / (srcMt.norm2() * targetMt.norm2());
		return sim;
	}
	
	/**
	 * Calculate similar between two words lists with weights by statistic way
	 * 
	 * @param sources Source words list
	 * @param sourceWeights Source words weight
	 * @param targets Target words list
	 * @param targetWeights Target words weight
	 * @return Similar score
	 */
	private double distanceStatistic(Collection<String> sources, Map<String, Double> sourceWeights, 
			Collection<String> targets, Map<String, Double> targetWeights)
	{
		double m = sources.size();
		double n = targets.size();
		DoubleMatrix similarMatrix = DoubleMatrix.zeros(sources.size(), targets.size());
		double sum1 = 0;
		int i = 0;
		int j = 0;
		for (String w1 : sources)
		{
			j = 0;
			double max = Double.NEGATIVE_INFINITY;
			for (String w2 : targets)
			{
				double s = similarMatrix.get(i, j);
				if (Double.compare(s, 0) == 0)
				{
					s = distance(w1, w2);
					similarMatrix.put(i, j, s);
				}
				max = Math.max(max, s);
				j++;
			}
			Double weight = sourceWeights == null ? null : sourceWeights.get(w1);
			if (weight != null)
			{
				m += weight - 1;
				max *= weight;
			}
			sum1 += max;
			i++;
		}
		double v1 = sum1 / m;
		
		double sum2 = 0;
		i = 0;
		j = 0;
		for (String w2 : targets)
		{
			i = 0;
			double max = Double.NEGATIVE_INFINITY;
			for (String w1 : sources)
			{
				double s = similarMatrix.get(i, j);
				if (Double.compare(s, 0) == 0)
				{
					s = distance(w1, w2);
					similarMatrix.put(i, j, s);
				}
				max = Math.max(max, s);
				i++;
			}
			Double weight = targetWeights == null ? null : targetWeights.get(w2);
			if (weight != null)
			{
				n += weight - 1;
				max *= weight;
			}
			sum2 += max;
			j++;
		}
		double v2 = sum2 / (double)n;
		return (v1 + v2) / 2.D;
	}

	/**
	 * Calculate specify word's nearest or relate words
	 * 
	 * @param word Specify word
	 * @return Nearest or relate words
	 */
	public Set<WordEntry> distance(String word)
	{
		return distance(word, config.topCount);
	}
	/**
	 * Calculate specify word's nearest or relate words
	 * 
	 * @param word Specify word
	 * @param topCount Result size
	 * @return Nearest or relate words
	 */
	public Set<WordEntry> distance(String word, int topCount)
	{
		int resultSize = FastMath.min(topCount, wordNodes.size());
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
	
	/**
	 * Get sentence feature
	 * @param words Sentence words
	 * @return matrix
	 */
	public DoubleMatrix getSentenceFeature(Collection<String> words)
	{
		DoubleMatrix center = null;
		for (String word : words)
		{
			WordNode node = wordNodes.get(word);
			if (node != null)
			{
				DoubleMatrix feature = node.feature;
				if (center == null)
				{
					center = feature.dup();
				}
				else
				{
					center.addi(feature);
				}
			}
		}
		return center;
	}
	
	/**
	 * Get sentence feature
	 * @param words Sentence words
	 * @param weights Word weights
	 * @return matrix
	 */
	public DoubleMatrix getSentenceFeature(Collection<String> words, Map<String, Double> weights)
	{
		DoubleMatrix center = null;
		for (String word : words)
		{
			WordNode node = wordNodes.get(word);
			if (node != null)
			{
				DoubleMatrix feature = node.feature.dup();
				Double weight = weights == null ? null : weights.get(word);
				if (weight != null)
				{
					feature.muli(weight);
				}
				if (center == null)
				{
					center = feature;
				}
				else
				{
					center.addi(feature);
				}
			}
		}
		return center;
	}
	
	/**
	 * Get specify word's vector
	 * 
	 * @param word Specify word
	 * @return Result vector
	 */
	public DoubleMatrix getWordVector(String word) {
		WordNode node = wordNodes.get(word);
		if (node == null)
		{
			return null;
		}
		return node.feature;
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
			log.error(e.getMessage(), e);
		}
		finally
		{
			IOUtils.closeStream(dos);
		}
	}

	/**
	 * Load model from single word
	 * 
	 * @param word Single word
	 * @param matrix Word feature
	 */
	public void loadModelFromWord(String word, DoubleMatrix matrix)
	{
		config.featureSize = matrix.length;
		wordNodes.put(word, new WordNode(word, matrix));
	}

	/**
	 * Load model from single word
	 * 
	 * @param word Single word
	 * @param attrs Word feature
	 */
	public void loadModelFromWord(String word, String attrs)
	{
		String[] features = attrs.split(" ");
		int featureSize = features.length;
		DoubleMatrix feature = new DoubleMatrix(featureSize);
		double len = 0;
		for (int f = 0; f < featureSize; f++)
		{
			double w = Double.parseDouble(features[f]);
			feature.put(f, w);
			len += w * w;
		}
		len = FastMath.sqrt(len);
		//feature.divi(len);
		wordNodes.put(word, new WordNode(word, feature));
		if (!parallelLoad)
		{
			config.featureSize = feature.length;
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
//				feature.divi(len);
				wordNodes.put(name, new WordNode(name, feature));
			}
			log.info("Succeed to read word2vec model. Word dictionary size " + wordNodes.size());
		}
		catch (Exception e)
		{
			log.error(e.getMessage(), e);
		}
		finally
		{
			IOUtils.closeStream(dis);
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

	public WordNode[] getUnigramTable()
	{
		return unigramTable;
	}

	public boolean isParallelLoad()
	{
		return parallelLoad;
	}

	public void setParallelLoad(boolean parallelLoad)
	{
		this.parallelLoad = parallelLoad;
		if (parallelLoad)
		{
			wordNodes = new ConcurrentHashMap<String, WordNode>(wordNodes);
		}
		else
		{
			wordNodes = new HashMap<String, WordNode>(wordNodes);
		}
	}

	public void setProgressReporter(ProgressReporter progressReporter)
	{
		this.progressReporter = progressReporter;
	}

	
}
