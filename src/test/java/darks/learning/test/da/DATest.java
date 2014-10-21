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
import org.junit.Test;

import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.da.DenoisingAutoEncoder;
import darks.learning.optimize.LearningOptimizer.OptimizeType;

public class DATest
{

	@Test
	public void testDA()
	{
		double[][] trainX = {
				{0, 1, 1, 0, 0, 0},
				{0, 0, 1, 0, 0, 0},
				{0, 1, 0, 0, 0, 0},
				{1, 0, 1, 0, 0, 0},
				{1, 1, 0, 0, 0, 0},
				{1, 1, 1, 0, 0, 0},
				{0, 0, 0, 1, 1, 1},
				{0, 0, 0, 1, 1, 0},
				{0, 0, 0, 1, 0, 1},
				{0, 0, 0, 0, 1, 1},
				{0, 0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0, 0}
			};
		
		DenoisingAutoEncoder da = new DenoisingAutoEncoder();
		da.config.setHiddenSize(64)
				.setMaxIterateCount(10000)
				.setMomentum(0.1)
				.setLossType(LossFunction.RECONSTRUCTION_CROSSENTROPY)
				.setUseAdaGrad(false)
				.setOptimizeType(OptimizeType.LINE_SEARCH);
		da.train(new DoubleMatrix(trainX));
		
		double[][] testX = {
				{0, 1, 1, 0, 0, 0},
				{0, 0, 1, 0, 0, 0},
				{0, 0, 0, 0, 1, 1},
				{0, 0, 0, 1, 0, 1}
			};
		DoubleMatrix ret = da.reconstruct(new DoubleMatrix(testX));
		System.out.println(ret.toString().replace(";", "\n"));
	}
	
}
