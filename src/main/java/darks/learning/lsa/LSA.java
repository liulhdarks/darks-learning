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
package darks.learning.lsa;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.math3.stat.Frequency;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;
import org.jblas.Singular;
import org.jblas.Solve;

public class LSA
{
	
	private Frequency freqCount = new Frequency();
	
	private Map<String, Frequency> countMap = new HashMap<String, Frequency>();
	
	Map<String, Integer> wordsIndex = null;

	Map<Integer, String> lineIndex = new HashMap<Integer, String>();
	
	public static void main(String[] args)
	{
		LSA lsa = new LSA();
		lsa.loadCorpus();
		DoubleMatrix A = lsa.calculateTFIDF();
		String[] sentence = "你好 今天 怎么 突然 常用 入口 点 进去 呢 想 流量 订单 都 看不到 着急".split(" ");
		DoubleMatrix vector = lsa.calTFIDF(sentence);
		System.out.println(vector);
		
		A = A.getRange(0, 500, 0, 500);
		vector = vector.getRange(0, 500, 0, 1);
		DoubleMatrix[] USV = Singular.fullSVD(A);
//		double[][] arg = new double[][]
//		{
//				{0.1, 0.2, 0, 0.4, 0.5},
//				{0.2, 0.1, 0.3, 0.1, 0.1},
//				{0, 0.3, 0.1, 0.2, 0.1},
//				{0.1, 0.4, 0.3, 0.2, 0},
//		};
		int k = 300;
		DoubleMatrix U = USV[0];
		DoubleMatrix S = USV[1];
		DoubleMatrix V = USV[2];
		U = U.getRange(0, U.rows, 0, k);
		S = S.getRange(0, k, 0, 1);
		V = V.getRange(0, k, 0, V.columns);
		DoubleMatrix Ak = U.mulRowVector(S).mmul(V);
		System.out.println(Ak.rows + " " + Ak.columns);

		DoubleMatrix vu = vector.transpose().mmul(U);
		System.out.println(S.rows + " " + S.columns);
		DoubleMatrix S1 = Solve.pinv(S);
		System.out.println(vu.rows + " " + vu.columns);
		System.out.println(S1.rows + " " + S1.columns);
		vector = vu.mmul(Solve.pinv(S));
		
		DoubleMatrix div = vector.transpose().mmul(Ak);
		
		DoubleMatrix sum = Ak.muli(Ak).columnSums();
		DoubleMatrix mt = MatrixFunctions.sqrt(sum);
		double sqrt = vector.norm2();
		mt.muli(sqrt);
		mt = div.div(mt);
		System.out.println(mt);
		int index = SimpleBlas.iamax(mt);
		System.out.println(lsa.lineIndex.get(index));
	}
	
	public void loadCorpus()
	{
		try
		{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("data/corpus_ali.txt"), "UTF-8"));
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				StringTokenizer token = new StringTokenizer(line, " \t");
				Set<String> set = new HashSet<String>();
				Frequency freq = countMap.get(line);
				if (freq == null)
				{
					freq = new Frequency();
					countMap.put(line, freq);
				}
				while (token.hasMoreTokens())
				{
					String word = token.nextToken();
					if (!set.contains(word))
					{
						freqCount.addValue(word);
						set.add(word);
					}
					freq.addValue(word);
				}
			}
			reader.close();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		
	}
	
	public DoubleMatrix calculateTFIDF()
	{
		int corpusCount = countMap.size();
		int wordsCount = freqCount.getUniqueCount();
		System.out.println(wordsCount + " " + corpusCount);
		DoubleMatrix result = new DoubleMatrix(wordsCount, corpusCount);
		int columnIndex = 0;
		int rowIndex = 0;
		wordsIndex = new HashMap<String, Integer>();
		for (Entry<String, Frequency> entryFreq : countMap.entrySet())
		{
			Frequency freq = entryFreq.getValue();
			int totalCount = (int) freq.getSumFreq();
			Iterator<Map.Entry<Comparable<?>, Long>> it = freq.entrySetIterator();
			while (it.hasNext())
			{
				Entry<Comparable<?>, Long> entry = it.next();
				String word = (String)entry.getKey();
				int wordCount = entry.getValue().intValue();
				int countInCorpus = (int)freqCount.getCount(word);
				double tf = (double)wordCount / (double)totalCount;
				double idf = Math.log((double)corpusCount / (double)(countInCorpus + 1));
				double tf_idf = tf * idf;
				Integer index = wordsIndex.get(word);
				if (index == null)
				{
					index = rowIndex++;
					wordsIndex.put(word, index);
				}
				result.put(index, columnIndex, tf_idf);
			}
			lineIndex.put(columnIndex, entryFreq.getKey());
			columnIndex++;
		}
		return result;
	}
	
	private DoubleMatrix calTFIDF(String[] sentence)
	{
		int corpusCount = countMap.size();
		int wordsCount = freqCount.getUniqueCount();
		DoubleMatrix result = new DoubleMatrix(wordsCount);
		int totalCount = sentence.length;
		Frequency freq = new Frequency();
		for (String s : sentence)
		{
			freq.addValue(s);
		}
		Iterator<Map.Entry<Comparable<?>, Long>> it = freq.entrySetIterator();
		while (it.hasNext())
		{
			Entry<Comparable<?>, Long> entry = it.next();
			String word = (String)entry.getKey();
			Integer index = wordsIndex.get(word);
			if (index == null)
			{
				System.out.println("cannot find " + word);
				continue;
			}
			int wordCount = entry.getValue().intValue();
			int countInCorpus = (int)freqCount.getCount(word);
			double tf = (double)wordCount / (double)totalCount;
			double idf = Math.log((double)corpusCount / (double)(countInCorpus + 1));
			double tf_idf = tf * idf;	
			result.put(index, tf_idf);
		}
		return result;
	}
	
}
