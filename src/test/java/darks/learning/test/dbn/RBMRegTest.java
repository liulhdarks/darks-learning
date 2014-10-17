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
package darks.learning.test.dbn;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.rbm.RBM;
import darks.learning.neuron.rbm.RBMConfig.LayoutType;
import darks.learning.optimize.LearningOptimizer.OptimizeType;
import darks.learning.regression.Regression;
import darks.learning.regression.SoftmaxRegression;

public class RBMRegTest
{

	@Test
	public void testRBMWithRegression()
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
		
		double[][] labels = {
				{1, 0},
				{1, 0},
				{1, 0},
				{1, 0},
				{1, 0},
				{1, 0},
				{0, 1},
				{0, 1},
				{0, 1},
				{0, 1},
				{0, 1},
				{0, 1},
			}; 
		
		RBM rbm1 = new RBM();
		rbm1.config.setHiddenSize(64)
				.setMaxIterateCount(10000)
				.setMomentum(0)
				.setLossType(LossFunction.RECONSTRUCTION_CROSSENTROPY)
				.setGibbsCount(1)
				.setLayoutType(LayoutType.BINARY)
				.setUseAdaGrad(false)
				.setOptimizeType(OptimizeType.LINE_SEARCH);
		rbm1.train(new DoubleMatrix(trainX));
		
		DoubleMatrix input2 = rbm1.propForward(new DoubleMatrix(trainX));
		RBM rbm2 = new RBM();
		rbm2.config.setHiddenSize(128)
				.setMaxIterateCount(10000)
				.setMomentum(0)
				.setLossType(LossFunction.RECONSTRUCTION_CROSSENTROPY)
				.setGibbsCount(1)
				.setLayoutType(LayoutType.BINARY)
				.setUseAdaGrad(false)
				.setOptimizeType(OptimizeType.LINE_SEARCH);
		rbm2.train(input2);
		
		DoubleMatrix input3 = rbm2.propForward(input2);
		
		Regression reg = new SoftmaxRegression();
		reg.config.setLearnRate(0.001)
					.setMaxIteratorCount(10000)
					.setRandomGradient(false)
					.setNormalized(true);
		reg.train(input3, new DoubleMatrix(labels));
		
		double[][] testX = {
				{0, 1, 1, 0, 0, 0},
				{0, 0, 1, 0, 0, 0},
				{0, 0, 0, 0, 1, 1},
				{0, 0, 0, 1, 0, 1},
				{1, 0, 0, 1, 0, 1},
				{1, 1, 0, 1, 0, 1},
			};

		DoubleMatrix inputReg = rbm1.propForward(new DoubleMatrix(testX));
		inputReg = rbm2.propForward(inputReg);
		DoubleMatrix result = reg.predict(inputReg);
		System.out.println(result.toString().replace(";", "\n"));
	}
	
}
