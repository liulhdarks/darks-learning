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

import static darks.learning.common.utils.MatrixHelper.svd;

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

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.Solve;

import darks.learning.common.utils.FreqCount;
import darks.learning.common.utils.MatrixHelper;

public class CopyOfLSA
{
	
	private FreqCount<String> freqCount = new FreqCount<String>();
	
	private Map<String, FreqCount<String>> countMap = new HashMap<String, FreqCount<String>>();

	static Map<String, Integer> wordsIndex = new HashMap<String, Integer>();

	static Map<Integer, String> lineIndex = new HashMap<Integer, String>();
	
	static TfIdf tfidf = new TfIdf();
    
    private static int K = 300;
    
    static DoubleMatrix preMatrix = null;
    
    static DoubleMatrix preNorm = null;
    
    static DoubleMatrix inverseS = null;
    
    static DoubleMatrix Uk = null;
	
	public static void main(String[] args)
	{
		CopyOfLSA lsa = new CopyOfLSA();
		lsa.loadCorpus();
		DoubleMatrix trainMatrix = lsa.initTrainData();
//		String[] sentence = "无法 打开 出售 中 宝贝 详情 页面".split(" ");
//		DoubleMatrix vector = lsa.calTFIDF(sentence);
//        System.out.println(vector.mean());
//		System.out.println(vector);
		
		trainMatrix = trainMatrix.getRange(0, 400, 0, 500);
        DoubleMatrix[] USV = svd(trainMatrix);
        
        DoubleMatrix U = USV[0];
        DoubleMatrix S = USV[1];
//        DoubleMatrix V = USV[2];
        Uk = U.getRange(0, U.rows, 0, K); //m*k
        DoubleMatrix Sk = S.getRange(0, K, 0, K); //k*k
//        DoubleMatrix Vk = V.getRange(0, K, 0, V.columns);//k*n

        inverseS = Solve.pinv(Sk);
        preMatrix = trainMatrix.transpose().mmul(Uk).mmul(inverseS); //n*k
        preNorm = MatrixHelper.sqrt(preMatrix.mul(preMatrix).rowSums());
        
        lsa.predict("无法 打开 出售 中 宝贝 详情 页面".split(" "));
        lsa.predict("刷新 订单 数量 还是 显示 错乱 旺旺 聊天 查看 买家 信誉 不了 以前 老 版本 没有 这种 问题".split(" "));
        lsa.predict("怎么 你们 千牛 软件 电脑 版 弹不出 聊天 窗口".split(" "));
        lsa.predict("为什么 手机 登陆 旺旺 聊天记录 其它 客服 接待 现实 号".split(" "));
        lsa.predict("电脑 登陆 时候 说 数据 不能 保存 系统盘 只有 一个盘 没有 分区".split(" "));
        lsa.predict("收不到 系统 提示信息 还有 设置 信息 声音 提示 怎么 会 没有".split(" "));
        lsa.predict("千 牛 电脑 端 怎么 登陆 打开 聊天 窗口 一直 响应 黑 然后 怎么回事".split(" "));
        lsa.predict("千 牛 电脑 端 怎么 登陆 打开 聊天 窗口 一直 响应 黑 然后 怎么回事 求 回应 已经 反馈 几次 现在 不了 点 开 就是".split(" "));
//		DoubleMatrix q = vector.transpose().mmul(U).mmul(Sinv);  //1*k
//		
//		DoubleMatrix dotVector = q.mmul(An.transpose());
//		double vnorm = vector.norm2();
//		
//		DoubleMatrix cosMt = dotVector.div(anorm.mul(vnorm));
//		System.out.println(cosMt);
//		int index = SimpleBlas.iamax(cosMt);
//        System.out.println("index:" + index);
//		System.out.println(lsa.lineIndex.get(index));
	}
	
	public void similar(String senetence, DoubleMatrix An, DoubleMatrix anorm, DoubleMatrix U, DoubleMatrix Sinv)
	{
	    String[] sentence = senetence.split(" ");
        DoubleMatrix vector = getSentenceVector(sentence);
        vector = vector.getRange(0, 400, 0, 1);
        DoubleMatrix q = vector.transpose().mmul(U).mmul(Sinv);  //1*k
        DoubleMatrix dotVector = q.mmul(An.transpose());
        double vnorm = vector.norm2();
        
        DoubleMatrix cosMt = dotVector.div(anorm.mul(vnorm));
//        System.out.println(cosMt);
        int index = SimpleBlas.iamax(cosMt);
        System.out.println("=========================================================");
        System.out.println("source:" + senetence);
        System.out.println("index:" + index);
        System.out.println("target:" + lineIndex.get(index));
	}
	
    public String predict(String[] words)
    {
        int column = predictIndex(words);
        String line = lineIndex.get(column);
        System.out.println(line);
        return line;
    }
    
    public int predictIndex(String[] words)
    {
        DoubleMatrix vector = getSentenceVector(words);
        vector = vector.getRange(0, 400, 0, 1);
        DoubleMatrix vectorMapping = vector.transpose().mmul(Uk).mmul(inverseS);  //1*k
        DoubleMatrix dotVector = vectorMapping.mmul(preMatrix.transpose());
        double norm = vector.norm2();
        DoubleMatrix cosMt = dotVector.div(preNorm.mul(norm));
        return SimpleBlas.iamax(cosMt);
    }
	
	public void loadCorpus()
	{
		try
		{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("corpus/corpus_ali_train.txt"), "UTF-8"));
			String line = null;
			while ((line = reader.readLine()) != null)
			{
			    line = line.trim();
				StringTokenizer token = new StringTokenizer(line, " \t");
				Set<String> set = new HashSet<String>();
				FreqCount<String> freq = countMap.get(line);
				if (freq == null)
				{
					freq = new FreqCount<String>();
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
					tfidf.addWord(line, word);
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
		int wordsCount = (int)freqCount.getUniqueCount();
		System.out.println(wordsCount + " " + corpusCount);
		DoubleMatrix result = new DoubleMatrix(wordsCount, corpusCount);
		int columnIndex = 0;
		int rowIndex = 0;
		for (Entry<String, FreqCount<String>> entryFreq : countMap.entrySet())
		{
			FreqCount<String> freq = entryFreq.getValue();
			int totalCount = (int) freq.totalCount();
			Iterator<Map.Entry<String, Long>> it = freq.entrySetIterator();
			while (it.hasNext())
			{
				Entry<String, Long> entry = it.next();
				String word = (String)entry.getKey();
				int wordCount = entry.getValue().intValue();
				int countInCorpus = freqCount.getValue(word).intValue();
				double tf = (double)wordCount / (double)totalCount;
				double idf = Math.log((double)corpusCount / (double)(countInCorpus + 1));
				double tf_idf2 = tf * idf;
				double tf_idf = tfidf.getTFIdf(entryFreq.getKey(), word);
				Integer index = wordsIndex.get(word);
				if (index == null)
				{
					index = rowIndex++;
					wordsIndex.put(word, index);
				}
				result.put(index, columnIndex, tf_idf);
//                System.out.print(tf_idf + " ");
			}
//            System.out.println();
			lineIndex.put(columnIndex, entryFreq.getKey());
			columnIndex++;
		}
		return result;
	}
    
    private DoubleMatrix initTrainData()
    {
        int corpusCount = (int)tfidf.getTotalSentenceCount();
        int wordsCount = (int)freqCount.getUniqueCount();
        System.out.println(wordsCount + " " + corpusCount);
        DoubleMatrix result = new DoubleMatrix(wordsCount, corpusCount);
        int columnIndex = 0;
        int rowIndex = 0;
        for (Entry<String, FreqCount<String>> entryFreq : tfidf.getSentenceMap().entrySet())
        {
            String sentence = entryFreq.getKey();
            FreqCount<String> freq = entryFreq.getValue();
            Iterator<Map.Entry<String, Long>> it = freq.entrySetIterator();
            while (it.hasNext())
            {
                Entry<String, Long> entry = it.next();
                String word = (String)entry.getKey();
                double tfIdfValue = tfidf.getTFIdf(sentence, word);
                Integer wordRowIndex = wordsIndex.get(word);
                if (wordRowIndex == null)
                {
                    wordRowIndex = rowIndex++;
                    wordsIndex.put(word, wordRowIndex);
                }
                result.put(wordRowIndex, columnIndex, tfIdfValue);
            }
            lineIndex.put(columnIndex, entryFreq.getKey());
            columnIndex++;
        }
        return result;
    }
	
	private DoubleMatrix calTFIDF(String[] sentence)
	{
		int corpusCount = countMap.size();
		int wordsCount = (int)freqCount.getUniqueCount();
		DoubleMatrix result = new DoubleMatrix(wordsCount);
		int totalCount = sentence.length;
		FreqCount<String> freq = new FreqCount<String>();
		for (String s : sentence)
		{
			freq.addValue(s);
		}
		Iterator<Entry<String, Long>> it = freq.entrySetIterator();
		while (it.hasNext())
		{
			Entry<String, Long> entry = it.next();
			String word = (String)entry.getKey();
			Integer index = wordsIndex.get(word);
			if (index == null)
			{
				System.out.println("cannot find " + word);
				continue;
			}
			int wordCount = entry.getValue().intValue();
			int countInCorpus = freqCount.getValue(word).intValue();
			double tf = (double)wordCount / (double)totalCount;
			double idf = Math.log((double)corpusCount / (double)(countInCorpus + 1));
			double tf_idf2 = tf * idf;	
            double tf_idf = tfidf.getTFIdf(freq, word);
			result.put(index, tf_idf);
		}
		return result;
	}
    
    private DoubleMatrix getSentenceVector(String[] sentence)
    {
        int wordsCount = (int)freqCount.getUniqueCount();
        DoubleMatrix result = new DoubleMatrix(wordsCount);
        FreqCount<String> freq = new FreqCount<String>();
        for (String s : sentence)
        {
            freq.addValue(s);
        }
        Iterator<Entry<String, Long>> it = freq.entrySetIterator();
        while (it.hasNext())
        {
            Entry<String, Long> entry = it.next();
            String word = (String)entry.getKey();
            Integer index = wordsIndex.get(word);
            if (index == null)
            {
                continue;
            }
            double tfIdf = tfidf.getTFIdf(freq, word);
            result.put(index, tfIdf);
        }
        return result;
    }
	
}
