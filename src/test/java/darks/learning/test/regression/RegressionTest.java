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

import darks.learning.model.ModelLoader;
import darks.learning.model.ModelSet;
import darks.learning.regression.LogisticRegression;
import darks.learning.regression.Regression;

public class RegressionTest
{

	@Test
	public void testLogisticRegression()
	{
//		double[][] input = new double[][]{
//				{3, 8},{5, 8},{7, 6},{8, 9},
//				{9, 6},{3, 9},{4, 7},{2, 6},
//				
//				{1, 3},{3, 2},{4, 1},{8, 4}, 
//				{3, 3},{4, 3},{9, 2},{6, 1},
//		};
//		double[][] output = new double[][]{
//				{1, 0},{1, 0},{1, 0},{1, 0},
//				{1, 0},{1, 0},{1, 0},{1, 0},
//				
//				{0, 1},{0, 1},{0, 1},{0, 1},
//				{0, 1},{0, 1},{0, 1},{0, 1},
//		};
		
		double[][] input = {
				{1, 1, 1, 0, 0, 0},
				{1, 0, 1, 0, 0, 0},
				{1, 1, 1, 0, 0, 0},
				{0, 0, 1, 1, 1, 0},
				{0, 0, 1, 1, 0, 0},
				{0, 0, 1, 1, 1, 0}
			};
			
		double[] output = {
				1, 1, 1,
				0, 0, 0
			}; 
		
		// test data
		double[][] test_X = {
			{1, 0, 1, 0, 0, 0},
			{0, 0, 1, 1, 1, 0},
			{1, 0, 1, 1, 1, 0},
		};
		ModelSet modelSet = ModelLoader.load(input, output);
		Regression reg = new LogisticRegression();
		reg.config.setLearnRate(0.001)
		.setMaxIteratorCount(100000)
		.setRandomGradient(true);
		reg.train(modelSet);
		DoubleMatrix result = reg.predict(new DoubleMatrix(test_X));
		System.out.println(result);
	}
	
}
