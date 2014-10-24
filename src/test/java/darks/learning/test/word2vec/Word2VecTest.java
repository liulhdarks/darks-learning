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
package darks.learning.test.word2vec;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

import org.junit.Test;

import darks.learning.corpus.Corpus;
import darks.learning.corpus.CorpusFilter;
import darks.learning.corpus.CorpusLoader;
import darks.learning.word2vec.Word2Vec;
import darks.learning.word2vec.Word2Vec.Word2VecType;

public class Word2VecTest
{

	@Test
	public void testTrain()
	{
		CorpusLoader loader = new CorpusLoader();
//		loader.addFilter(new CorpusFilter()
//		{
//			@Override
//			public boolean filter(String s)
//			{
//				return s.length() <= 1;
//			}
//		});
		loader.addStopwords(new File("corpus/dic/lex-stopword.lex"));
		loader.addStopwords(new File("corpus/dic/lex-stopword1.lex"));
		Corpus corpus = loader.loadFromFile(new File("corpus/corpus_ali.txt"));
		
		Word2Vec word2vec = new Word2Vec();
		word2vec.config.setTrainType(Word2VecType.CBOW)
						.setFeatureSize(100)
						.setMinVocabCount(0)
						.setWindow(5)
						.setNegative(0);
		word2vec.train(corpus);
		word2vec.saveModel(new File("test/test_ali.model"));
	}

	@Test
	public void testDistance()
	{
		Word2Vec vec = new Word2Vec();
		vec.loadModel(new File("test/test_ali.model"));
		System.out.println(vec.distance("版本"));
		System.out.println(vec.distance("分流"));
		double sim = vec.distance("计算机", "电脑");
		System.out.println(sim);
		List<String> sources = new LinkedList<String>(); //我国 自行 研制 了 功能 强大 的 机群 操作系统 
		List<String> targets = new LinkedList<String>();
		sources.add("希望");sources.add("人性化");
		targets.add("分流");targets.add("开关");
		sim = vec.distance(sources, targets);
		System.out.println(sim);
	}
	
}
