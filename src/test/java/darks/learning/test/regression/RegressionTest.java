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
package darks.learning.test.regression;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import darks.learning.classifier.regression.LogisticRegression;
import darks.learning.classifier.regression.Regression;
import darks.learning.classifier.regression.SoftmaxRegression;
import darks.learning.model.ModelLoader;
import darks.learning.model.ModelSet;

public class RegressionTest
{

	@Test
	public void testLogisticRegression()
	{
		
		double[][] trainX = {
				{0, 1, 1, 0, 0, 0},
				{1, 0, 1, 0, 0, 0},
				{1, 1, 1, 0, 0, 0},
				{0, 0, 0, 1, 1, 1},
				{0, 0, 0, 1, 1, 0},
				{0, 0, 0, 1, 0, 1}
			};
			
		double[] output = {
				0, 0, 0,
				1, 1, 1
			}; 
		
		// test data
		double[][] testX = {
			{1, 0, 1, 0, 0, 0},
			{1, 1, 1, 0, 0, 0},
			{0, 0, 1, 1, 0, 1},
			{1, 0, 0, 1, 1, 1}
		};
		ModelSet modelSet = ModelLoader.load(trainX, output);
		Regression reg = new LogisticRegression();
		reg.config.setLearnRate(0.001)
					.setMaxIteratorCount(100000)
					.setRandomGradient(false)
					.setNormalized(true)
					.setUseAdaGrad(false);
		reg.train(modelSet);
		DoubleMatrix result = reg.predict(new DoubleMatrix(testX));
		System.out.println(result.toString().replace(";", "\n"));
	}
	

	@Test
	public void testSoftmaxRegression()
	{
		
		double[][] trainX = {
				{0, 1, 1, 0, 0, 0, 0, 0, 0},
				{1, 0, 1, 0, 0, 0, 0, 0, 0},
				{1, 1, 1, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 1, 1, 1, 0, 0, 0},
				{0, 0, 0, 1, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 0, 1, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 1, 1, 1},
				{0, 0, 0, 0, 0, 0, 1, 1, 0},
				{0, 0, 0, 0, 0, 0, 1, 0, 1},
			};
			
		int[] output = {
				0, 0, 0,
				1, 1, 1,
				2, 2, 2
			}; 
		
//		double[][] output = {
//				{1, 0, 0},
//				{1, 0, 0},
//				{1, 0, 0},
//				{0, 0, 1},
//				{0, 0, 1},
//				{0, 0, 1},
//				{0, 1, 0},
//				{0, 1, 0},
//				{0, 1, 0},
//			}; 
		
		// test data
		double[][] testX = {
			{1, 0, 1, 0, 0, 0, 0, 0, 0},
			{1, 1, 1, 0, 0, 0, 1, 0, 0},
			{0, 0, 1, 1, 0, 1, 0, 0, 0},
			{1, 0, 0, 1, 1, 1, 0, 0, 0},
			{0, 0, 0, 1, 0, 0, 1, 1, 0},
			{0, 0, 0, 0, 0, 1, 0, 1, 1},
		};
		ModelSet modelSet = ModelLoader.load(trainX, output);
		Regression reg = new SoftmaxRegression();
		reg.config.setLearnRate(0.001)
					.setMaxIteratorCount(500000)
					.setRandomGradient(false)
					.setNormalized(true)
					.setUseAdaGrad(false);
		reg.train(modelSet);
		DoubleMatrix result = reg.predict(new DoubleMatrix(testX));
		System.out.println(result.toString().replace(";", "\n"));
	}
	
}
