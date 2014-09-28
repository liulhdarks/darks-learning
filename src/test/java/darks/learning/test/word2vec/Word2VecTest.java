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
		loader.addFilter(new CorpusFilter()
		{
			@Override
			public boolean filter(String s)
			{
				return s.length() <= 1;
			}
		});
		loader.addStopwords(new File("corpus/dic/lex-stopword.lex"));
		loader.addStopwords(new File("corpus/dic/lex-stopword1.lex"));
		Corpus corpus = loader.loadFromFile(new File("corpus/corpus.txt"));
		
		Word2Vec word2vec = new Word2Vec();
		word2vec.config.setTrainType(Word2VecType.CBOW)
						.setFeatureSize(100)
						.setMinVocabCount(0)
						.setNegative(5);
		word2vec.train(corpus);
		word2vec.saveModel(new File("test/test.model"));
	}

	@Test
	public void testDistance()
	{
		Word2Vec vec = new Word2Vec();
		vec.loadModel(new File("test/test.model"));
        System.out.println(vec.getWordVector("国家队").toString());
        System.out.println(vec.getWordVector("央视").toString());
		System.out.println(vec.distance("计算机"));
		System.out.println(vec.distance("学习"));
		System.out.println(vec.distance("研究"));
		System.out.println(vec.distance("服务器"));
		System.out.println(vec.distance("毛泽东"));
		System.out.println(vec.distance("飞机"));
		System.out.println(vec.distance("国家"));
		double sim = vec.distance("计算机", "电脑");
		System.out.println(sim);
	}
	
}
