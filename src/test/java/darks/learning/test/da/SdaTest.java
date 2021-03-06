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
package darks.learning.test.da;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Test;

import darks.learning.neuron.sda.StackedDenosingAutoEncoder;

public class SdaTest
{

	@Test
	public void testRBMWithRegression()
	{
		
		double[][] trainX = {
				{0, 0.9, 0.8, 0, 0, 0, 0, 0, 0},
				{0, 0, 1, 0, 0, 0, 0, 0, 0},
				{0, 1, 0, 0, 0, 0, 0, 0, 0},
				{1, 0, 1, 0, 0, 0, 0, 0, 0},
				{1, 1, 0, 0, 0, 0, 0, 0, 0},
				{1, 1, 1, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 1, 1, 1, 0, 0, 0},
				{0, 0, 0, 1, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 0, 1, 0, 0, 0},
				{0, 0, 0, 0, 1, 1, 0, 0, 0},
				{0, 0, 0, 0, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 1, 1, 1},
				{0, 0, 0, 0, 0, 0, 1, 1, 0},
				{0, 0, 0, 0, 0, 0, 1, 0, 1},
				{0, 0, 0, 0, 0, 0, 0, 1, 1},
				{0, 0, 0, 0, 0, 0, 0, 1, 0},
				{0, 0, 0, 0, 0, 0, 1, 0, 0}
			};
		
		double[][] labels = {
				{1, 0, 0},
				{1, 0, 0},
				{1, 0, 0},
				{1, 0, 0},
				{1, 0, 0},
				{1, 0, 0},
				{0, 1, 0},
				{0, 1, 0},
				{0, 1, 0},
				{0, 1, 0},
				{0, 1, 0},
				{0, 1, 0},
				{0, 0, 1},
				{0, 0, 1},
				{0, 0, 1},
				{0, 0, 1},
				{0, 0, 1},
				{0, 0, 1},
			}; 
		
		
		StackedDenosingAutoEncoder sda = new StackedDenosingAutoEncoder();
		sda.config.setHiddenLayouts(new int[]{32, 64})
					.setNormalized(true)
					.setUseSample(false);
		sda.trainBatch(new DoubleMatrix(trainX), new DoubleMatrix(labels));
		
		double[][] testX = {
				{1, 1, 0, 0, 1, 0, 0, 0, 0},
				{0, 1, 1, 0, 0, 0, 0, 0, 0},
				{0, 0, 1, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 1, 1, 0, 0, 0},
				{0, 0, 0, 1, 0, 1, 0, 0, 0},
				{1, 0, 0, 1, 0, 1, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 1, 0, 1},
				{0, 0, 0, 0, 0, 0, 0, 1, 1},
				{0, 0, 0, 0, 0, 0, 1, 1, 1},
			};
		DoubleMatrix result = sda.predict(new DoubleMatrix(testX));
		System.out.println(result.toString("%f", "[\n", "\n]", ", ", "\n"));
		for (int i = 0; i < result.rows; i++)
		{
			System.out.println(SimpleBlas.iamax(result.getRow(i)));
		}
	}
	
}
