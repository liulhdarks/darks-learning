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
package darks.learning.test.cluster;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterPoint;
import darks.learning.cluster.hierarchy.DynamicalCluster;
import darks.learning.corpus.Corpus;
import darks.learning.corpus.CorpusLoader;
import darks.learning.distance.Distance;
import darks.learning.topic.plsa.ProbabilityLSA;
import darks.learning.word2vec.Word2Vec;

public class ClusterTest
{

	@Test
	public void testDynamicalCluster()
	{
//		RecordDistance distance = new RecordDistance("test/train_data.model");
		RecordLsaDistance distance = new RecordLsaDistance("corpus/train_data.txt");
		DynamicalCluster<TestRecord> cluster = new DynamicalCluster<TestRecord>(distance);
		
		try
		{
			BufferedReader reader = new BufferedReader(new FileReader("corpus/train_data.txt"));
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				line = line.trim();
				if ("".equals(line))
				{
					continue;
				}
				String[] words = line.split(" ");
				cluster.clusterIncrement(new TestRecord(line, Arrays.asList(words)));
			}
			cluster.optimizeCluster();
			for (Cluster<TestRecord> center : cluster.getClusters())
			{
				System.out.println("Cluster:" + center.getCenter().getObject().title);
				for (ClusterPoint<TestRecord> point : center.getPoints())
				{
					System.out.println("Point:" + point.getObject().title + " " + point.getSimilar());
				}
				System.out.println();
			}
			reader.close();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		
	}
	
	
	
	class RecordDistance implements Distance<TestRecord>
	{
		
		Word2Vec word2Vec;
		
		public RecordDistance(String modelPath)
		{
			word2Vec = new Word2Vec();
			word2Vec.loadModel(new File(modelPath));
		}

		@Override
		public double distance(TestRecord a, TestRecord b)
		{
			return word2Vec.distance(a.words, b.words);
		}

	}
	
	
	
	class RecordLsaDistance implements Distance<TestRecord>
	{
		
		ProbabilityLSA plsa;
		
		public RecordLsaDistance(String modelPath)
		{
			CorpusLoader loader = new CorpusLoader(Corpus.TYPE_TF_IDF);
	        File file = new File(modelPath);
	        Corpus corpus = loader.loadFromFile(file, "UTF-8");
	        plsa = new ProbabilityLSA(100);
	        plsa.setIterNumber(20);
	        plsa.train(corpus);
		}

		@Override
		public double distance(TestRecord a, TestRecord b)
		{
			return plsa.distanceDocuments(a.title, b.title);
		}

	}
	
	class TestRecord
	{
		String title;
		
		List<String> words;
		

		public TestRecord(String title, List<String> words)
		{
			super();
			this.title = title;
			this.words = words;
		}

		@Override
		public int hashCode()
		{
			final int prime = 31;
			int result = 1;
			result = prime * result + getOuterType().hashCode();
			result = prime * result + ((title == null) ? 0 : title.hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj)
		{
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			TestRecord other = (TestRecord) obj;
			if (!getOuterType().equals(other.getOuterType()))
				return false;
			if (title == null)
			{
				if (other.title != null)
					return false;
			}
			else if (!title.equals(other.title))
				return false;
			return true;
		}

		private ClusterTest getOuterType()
		{
			return ClusterTest.this;
		}

		@Override
		public String toString()
		{
			return "TestRecord [title=" + title + ", words=" + words + "]";
		}
		
	}
	
}
