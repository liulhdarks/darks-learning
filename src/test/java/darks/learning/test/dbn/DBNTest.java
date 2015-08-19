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
import org.jblas.SimpleBlas;
import org.junit.Test;

import darks.learning.eval.Evaluation;
import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.dbn.DBN;
import darks.learning.neuron.dbn.DBNConfig.FinetuneType;
import darks.learning.neuron.rbm.RBMConfig.LayoutType;
import darks.learning.optimize.LearningOptimizer.OptimizeType;

public class DBNTest
{

	@Test
	public void testRBMWithRegression()
	{
//		double[][] trainX = {
//				{0, 1, 1, 0, 0, 0},
//				{0, 0, 1, 0, 0, 0},
//				{0, 1, 0, 0, 0, 0},
//				{1, 0, 1, 0, 0, 0},
//				{1, 1, 0, 0, 0, 0},
//				{1, 1, 1, 0, 0, 0},
//				{0, 0, 0, 1, 1, 1},
//				{0, 0, 0, 1, 1, 0},
//				{0, 0, 0, 1, 0, 1},
//				{0, 0, 0, 0, 1, 1},
//				{0, 0, 0, 0, 1, 0},
//				{0, 0, 0, 1, 0, 0}
//			};
//		
//		double[][] labels = {
//				{1, 0},
//				{1, 0},
//				{1, 0},
//				{1, 0},
//				{1, 0},
//				{1, 0},
//				{0, 1},
//				{0, 1},
//				{0, 1},
//				{0, 1},
//				{0, 1},
//				{0, 1},
//			}; 
		
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
		
		
		DBN dbn = new DBN();
//		dbn.config.setHiddenLayouts(new int[]{32, 16})
//					.setNormalized(true)
//					.setUseSample(false)
//					.setHiddenLayoutType(LayoutType.BINARY)
//					.setFineTuneIterateCount(50000);
		dbn.config.setHiddenLayouts(new int[]{32, 24})
			.setNormalized(true)
			.setUseSample(false)
			.setHiddenLayoutType(LayoutType.BINARY)
//			.setOptimizeTypes(OptimizeType.LINE_SEARCH)
			.setHiddensIterateCount(5000)
			.setFinetuneType(FinetuneType.MLP)
			.setFineTuneOutputLayerSize(3)
//			.setFineTuneHiddenLayouts(new int[]{16, 8})
			.setFineTuneIterateCount(50000)
			.setFineTuneLearnRate(0.1)
			.setFineTuneLossType(LossFunction.MSE);
		dbn.trainBatch(new DoubleMatrix(trainX), new DoubleMatrix(labels));
		
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
		

		double[][] testLabels = {
				{1, 0, 0},
				{1, 0, 0},
				{1, 0, 0},
				{0, 1, 0},
				{0, 1, 0},
				{0, 1, 0},
				{0, 0, 1},
				{0, 0, 1},
				{0, 0, 1}
			};
		DoubleMatrix result = dbn.predict(new DoubleMatrix(testX));
		System.out.println(result.toString("%f", "[\n", "\n]", ", ", "\n"));
		for (int i = 0; i < result.rows; i++)
		{
			System.out.println(SimpleBlas.iamax(result.getRow(i)));
		}
		Evaluation eval = new Evaluation();
		eval.eval(result, new DoubleMatrix(testLabels));
		System.out.println(eval.f1());
	}
	
}
