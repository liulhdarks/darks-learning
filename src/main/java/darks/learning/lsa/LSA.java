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

import org.apache.commons.math3.stat.Frequency;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;
import org.jblas.Singular;
import org.jblas.Solve;

import darks.learning.common.utils.MatrixHelper;

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
//      String[] sentence = "无法 打开 出售 中 宝贝 详情 页面".split(" ");
//      DoubleMatrix vector = lsa.calTFIDF(sentence);
//        System.out.println(vector.mean());
//      System.out.println(vector);
        
//        A = A.getRange(0, 400, 0, 500);
        DoubleMatrix[] USV = svd(A);
        int k = 300;
        DoubleMatrix U = USV[0];
        DoubleMatrix S = USV[1];
        DoubleMatrix V = USV[2];
        U = U.getRange(0, U.rows, 0, k);
        S = S.getRange(0, k, 0, k);
        V = V.getRange(0, k, 0, V.columns);

        DoubleMatrix Sinv = Solve.pinv(S);
        DoubleMatrix An = A.transpose().mmul(U).mmul(Sinv); //n*k
        DoubleMatrix anorm = MatrixHelper.sqrt(An.mul(An).rowSums());
		
		lsa.similar("无法 打开 出售 中 宝贝 详情 页面", An, anorm, U, Sinv);
        lsa.similar("刷新 订单 数量 还是 显示 错乱 旺旺 聊天 查看 买家 信誉 不了 以前 老 版本 没有 这种 问题", An, anorm, U, Sinv);
        lsa.similar("怎么 你们 千牛 软件 电脑 版 弹不出 聊天 窗口", An, anorm, U, Sinv);
        lsa.similar("为什么 手机 登陆 旺旺 聊天记录 其它 客服 接待 现实 号", An, anorm, U, Sinv);
        lsa.similar("电脑 登陆 时候 说 数据 不能 保存 系统盘 只有 一个盘 没有 分区", An, anorm, U, Sinv);
        lsa.similar("收不到 系统 提示信息 还有 设置 信息 声音 提示 怎么 会 没有", An, anorm, U, Sinv);
        lsa.similar("千 牛 电脑 端 怎么 登陆 打开 聊天 窗口 一直 响应 黑 然后 怎么回事", An, anorm, U, Sinv);
        lsa.similar("千 牛 电脑 端 怎么 登陆 打开 聊天 窗口 一直 响应 黑 然后 怎么回事 求 回应 已经 反馈 几次 现在 不了 点 开 就是", An, anorm, U, Sinv);
	}
    
    public void similar(String senetence, DoubleMatrix An, DoubleMatrix anorm, DoubleMatrix U, DoubleMatrix Sinv)
    {
        String[] sentence = senetence.split(" ");
        DoubleMatrix vector = calTFIDF(sentence);
//        vector = vector.getRange(0, 400, 0, 1);
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
	
	public void loadCorpus()
	{
		try
		{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("corpus/corpus_ali_train.txt"), "UTF-8"));
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
