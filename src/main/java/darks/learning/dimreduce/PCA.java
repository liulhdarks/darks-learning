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
package darks.learning.dimreduce;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.Eigen;

/**
 * Principal Component Analysis
 * 
 * @author Darks.Liu
 *
 */
public class PCA
{
	
	/**
	 * Reduce matrix dimension
	 * 
	 * @param source Source matrix
	 * @param dimension Target dimension
	 * @return Target matrix
	 */
	public static DoubleMatrix dimensionReduction(DoubleMatrix source, int dimension)
	{
		//C=X*X^t / m
		DoubleMatrix covMatrix = source.mmul(source.transpose()).div(source.columns);
		ComplexDoubleMatrix eigVal = Eigen.eigenvalues(covMatrix);
		ComplexDoubleMatrix[] eigVectorsVal = Eigen.eigenvectors(covMatrix);
		ComplexDoubleMatrix eigVectors = eigVectorsVal[0];
		//Sort sigen vector from big to small by eigen values 
		List<PCABean> beans = new ArrayList<PCA.PCABean>();
		for (int i = 0; i < eigVectors.columns; i++)
		{
			beans.add(new PCABean(eigVal.get(i).real(), eigVectors.getColumn(i)));
		}
		Collections.sort(beans);
		DoubleMatrix newVec = new DoubleMatrix(dimension, beans.get(0).vector.rows);
		for (int i = 0; i < dimension; i++)
		{
			ComplexDoubleMatrix dm = beans.get(i).vector;
			DoubleMatrix real = dm.getReal();
			newVec.putRow(i, real);
		}
		return newVec.mmul(source);
	}
	
	static class PCABean implements Comparable<PCABean>
	{
		double eigenValue;
		
		ComplexDoubleMatrix vector;
		
		public PCABean(double eigenValue, ComplexDoubleMatrix vector)
		{
			super();
			this.eigenValue = eigenValue;
			this.vector = vector;
		}



		@Override
		public int compareTo(PCABean o)
		{
			return Double.compare(o.eigenValue, eigenValue);
		}



		@Override
		public String toString()
		{
			return "PCABean [eigenValue=" + eigenValue + ", vector=" + vector + "]";
		}
		
	}
}
