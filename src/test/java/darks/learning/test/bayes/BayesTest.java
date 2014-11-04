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

import org.junit.Test;

import darks.learning.classifier.bayes.NativeBayes;
import darks.learning.corpus.Documents;

public class BayesTest
{

	@Test
	public void testNativeBayes()
	{
		File input = new File("corpus/train_data.txt");
		File labels = new File("corpus/train_labels.txt");
		Documents docs;
		try
		{
			docs = Documents.loadFromFile(input, labels, "UTF-8");
			NativeBayes bayes = new NativeBayes();
			bayes.config.setLogLikelihood(true)
						.setModelType(NativeBayes.BINAMIAL);
			bayes.train(docs);
			System.out.println(bayes.predict("显示 卖家版旺旺 状态 账号 分流 登录 手机 "));
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
	
}
