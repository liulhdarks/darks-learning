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
package darks.learning.test.eval;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import darks.learning.eval.Evaluation;

public class EvalTest
{

	@Test
	public void testEval()
	{
		double[][] readArg = new double[][]{
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 1, 0, 0},
				{0, 1, 0, 0},
				{0, 1, 0, 0},
				{0, 1, 0, 0}
		};
		double[][] guessArg = new double[][]{
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
				{0, 1, 0, 0},
				{0, 1, 0, 0},
				{0, 1, 0, 0}
		};
		Evaluation eval = new Evaluation();
		DoubleMatrix real = new DoubleMatrix(readArg);
		DoubleMatrix guesses = new DoubleMatrix(guessArg);
		eval.eval(real, guesses);
		System.out.println(eval.f1());
		System.out.println(eval.precision());
		System.out.println(eval.accuracy());
		System.out.println(eval.recall());
		System.out.println(eval.status());
		
	}
	
}
