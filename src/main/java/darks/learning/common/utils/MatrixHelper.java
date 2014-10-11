/**
 * 
 * Copyright 2014 The Darks Learning Project (Liu lihua)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */
package darks.learning.common.utils;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import darks.learning.common.distribution.Distributions;
import darks.learning.common.rand.RandomFunction;

public class MatrixHelper
{

	public static DoubleMatrix log(DoubleMatrix mt)
	{
		return MatrixFunctions.log(mt);
	}

	public static DoubleMatrix exp(DoubleMatrix mt)
	{
		return MatrixFunctions.exp(mt);
	}

	public static DoubleMatrix tanh(DoubleMatrix mt)
	{
		return MatrixFunctions.tanh(mt);
	}

	public static DoubleMatrix pow(DoubleMatrix mt, double e)
	{
		return MatrixFunctions.pow(mt, e);
	}

	public static DoubleMatrix sigmoid(DoubleMatrix mt)
	{
		DoubleMatrix ones = DoubleMatrix.ones(mt.rows, mt.columns);
		return ones.div(ones.add(exp(mt.neg())));
	}

	public static DoubleMatrix oneMinus(DoubleMatrix mt)
	{
		return DoubleMatrix.ones(mt.rows, mt.columns).sub(mt);
	}

	public static DoubleMatrix softmax(DoubleMatrix mt)
	{
		DoubleMatrix max = mt.rowMaxs();
		DoubleMatrix diff = MatrixFunctions.exp(mt.subColumnVector(max));
		diff.diviColumnVector(diff.rowSums());
		return diff;
	}

	public static DoubleMatrix gaussion(int rows, int columns)
	{
		return gaussioni(new DoubleMatrix(rows, columns));
	}

	public static DoubleMatrix gaussioni(DoubleMatrix mt)
	{
		for (int i = 0; i < mt.rows; i++)
		{
			for (int j = 0; j < mt.columns; j++)
			{
				mt.put(i, j, Distributions.normal());
			}
		}
		return mt;
	}

	public static DoubleMatrix binomial(DoubleMatrix p, RandomFunction rng)
	{
		DoubleMatrix ret = new DoubleMatrix(p.rows, p.columns);
		for (int i = 0; i < ret.length; i++)
		{
			ret.put(i, (rng.randDouble() < p.get(i) ? 1 : 0));
		}
		return ret;
	}
}
