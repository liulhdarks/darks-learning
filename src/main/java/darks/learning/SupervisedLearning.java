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
package darks.learning;

import org.jblas.DoubleMatrix;

/**
 * Supervised machine learning
 * 
 * @author Darks.Liu
 *
 */
public interface SupervisedLearning
{

	/**
	 * Batch train model set,Train model until converge
	 * 
	 * @param input Model input
	 * @param output Model output
	 */
	public void trainBatch(DoubleMatrix input, DoubleMatrix output);

	/**
	 * Train model set
	 * 
	 * @param iterateNumber Current iteration number
	 * @param input Model input
	 * @param output Model output
	 * @return Loss value
	 */
	public double train(int iterateNumber, DoubleMatrix input, DoubleMatrix output);
	
	
	
	/**
	 * Predict model result by input matrix
	 * 
	 * @param input Input matrix
	 * @return Result label matrix
	 */
	public DoubleMatrix predict(DoubleMatrix input);
}
