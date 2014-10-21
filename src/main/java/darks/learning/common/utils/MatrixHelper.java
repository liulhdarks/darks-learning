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

import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import darks.learning.common.distribution.Distributions;
import darks.learning.common.rand.RandomFunction;

public class MatrixHelper
{
	public static DoubleMatrix max(double min, DoubleMatrix matrix)
	{
		for (int i = 0; i < matrix.length; i++)
			matrix.put(i, Math.max(min, matrix.get(i)));
		return matrix;
	}

	public static DoubleMatrix log(DoubleMatrix mt)
	{
		return MatrixFunctions.log(mt);
	}

	public static DoubleMatrix abs(DoubleMatrix mt)
	{
		return MatrixFunctions.abs(mt);
	}

	public static DoubleMatrix exp(DoubleMatrix mt)
	{
		return MatrixFunctions.exp(mt);
	}

	public static DoubleMatrix sqrt(DoubleMatrix mt)
	{
		return MatrixFunctions.sqrt(mt);
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

	public static DoubleMatrix gaussion(DoubleMatrix mean, double sd)
	{
		DoubleMatrix result = new DoubleMatrix(mean.rows, mean.columns);
		for (int i = 0; i < result.rows; i++)
		{
			for (int j = 0; j < result.columns; j++)
			{
				result.put(i, j, Distributions.normal(mean.get(i, j), FastMath.sqrt(sd)));
			}
		}
		return result;
	}

	public static DoubleMatrix gaussion(DoubleMatrix mean, DoubleMatrix variance)
	{
		DoubleMatrix std = sqrt(variance);
		for (int i = 0; i < variance.length; i++)
		{
			if (variance.get(i) <= 0)
			{
				variance.put(i, 1e-4);
			}
		}

		DoubleMatrix result = new DoubleMatrix(mean.rows, mean.columns);
		for (int i = 0; i < result.rows; i++)
		{
			for (int j = 0; j < result.columns; j++)
			{
				result.put(i, j, Distributions.normal(mean.get(i, j), std.get(j)));
			}
		}
		return result;
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

	public static int binomial(double p, RandomFunction rng)
	{
		return rng.randDouble() < p ? 1 : 0;
	}

	public static DoubleMatrix columnVariance(DoubleMatrix input)
	{
		DoubleMatrix columnMeans = input.columnMeans();
		DoubleMatrix ret = new DoubleMatrix(1, columnMeans.columns);
		for (int i = 0; i < ret.columns; i++)
		{
			DoubleMatrix column = input.getColumn(i);
			double variance = StatUtils.variance(column.toArray(), columnMeans.get(i));
			if (variance == 0)
				variance = 1e-6;
			ret.put(i, variance);
		}
		return ret;
	}

	public static DoubleMatrix concatVector(DoubleMatrix target, DoubleMatrix vector)
	{
		DoubleMatrix result = new DoubleMatrix(target.rows, target.columns + vector.length);
		for (int i = 0; i < target.rows; i++)
		{
			DoubleMatrix row = target.getRow(i);
			row = DoubleMatrix.concatHorizontally(row, vector.transpose());
			result.putRow(i, row);
		}
		return result;
	}
}
