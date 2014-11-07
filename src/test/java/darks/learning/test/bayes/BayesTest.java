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
package darks.learning.test.bayes;

import java.io.File;
import java.io.IOException;
import java.util.Map.Entry;

import org.junit.Test;

import darks.learning.classifier.bayes.NaiveBayes;
import darks.learning.corpus.Documents;

public class BayesTest
{

	@Test
	public void testNaiveBayes()
	{
		File input = new File("corpus/train_data.txt");
		File labels = new File("corpus/train_labels.txt");
		Documents docs;
		try
		{
			docs = Documents.loadFromFile(input, labels, "UTF-8");
			NaiveBayes bayes = new NaiveBayes();
			bayes.config.setLogLikelihood(true)
						.setModelType(NaiveBayes.BINAMIAL);
			bayes.train(docs);
			int count = 0;
			for (Entry<String, String> entry : docs.getDocsMap().entrySet())
			{
				String classify = bayes.predict(entry.getKey());
				if (!classify.equals(entry.getValue()))
				{
					System.out.println("QA:" + entry.getKey() + " output:" + classify + " expect:" + entry.getValue());
				}
				else
				{
					count++;
				}
			}
			System.out.println("Accurancy:" + (float)count / (float)docs.getDocsMap().size());
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
	
}
