package darks.learning.test.math;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import darks.learning.common.utils.MatrixHelper;

public class MathTest
{

	@Test
	public void testSVD()
	{
		double[][] testData = { 
				{ 36, 49, 47, 11 }, 
				{ 2, 68, 27, 42 }, 
				{ 42, 25, 38, 3 } 
			};
		RealMatrix matrix = MatrixUtils.createRealMatrix(testData);
		SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
		System.out.println(svd.getS());
		System.out.println(svd.getV());
		System.out.println(svd.getU());
		
		double[][] data1 = { 
				{ 1, 2, 3}, 
				{ 4, 5, 6}, 
				{ 7, 8, 9} 
			};
		double[] blas = { 
				1, 2, 3
			};
		DoubleMatrix dmatrix = new DoubleMatrix(data1);
		DoubleMatrix bmatrix = new DoubleMatrix(blas);
		System.out.println(DoubleMatrix.concatHorizontally(dmatrix, bmatrix));
	}

	@Test
	public void testMatrix()
	{
		double[][] input = { 
				{1, 1, 0, 1},
				{1, 0, 1, 1},
				{1, 1, 1, 1},
			};
		double[][] weight = { 
				{1, 0, 0},
				{1, 1, 0},
				{0, 1, 1},
				{1, 0, 1},
			};
		double[][] output = { 
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			};
		DoubleMatrix im = new DoubleMatrix(input);
		DoubleMatrix wm = new DoubleMatrix(weight);
		DoubleMatrix outm = new DoubleMatrix(output);
		System.out.println(im.mmul(wm));
		
	}

	@Test
	public void testMatrixMul()
	{
		double[][] input = { 
				{1, 2, 3},
				{3, 4, 5}
			};
		double[][] weight = { 
				{0.1},
				{0.2},
				{0.3}
			};
		DoubleMatrix im = new DoubleMatrix(input);
		DoubleMatrix wm = new DoubleMatrix(weight);
		//System.out.println(im.mul(wm));
		System.out.println(MatrixHelper.sigmoid(im.mmul(wm)));
	}

	@Test
	public void testMatrixSubAdd()
	{
		double[][] input = { 
				{1, 2, 3},
				{3, 4, 5}
			};
		double[] bias = { 
				1, 2, 1
			};
		DoubleMatrix im = new DoubleMatrix(input);
		DoubleMatrix wm = new DoubleMatrix(bias);
		//System.out.println(im.mul(wm));
		System.out.println(wm.sub(im.columnMeans()));
	}
}
