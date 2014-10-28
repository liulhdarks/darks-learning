package darks.learning.test.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.Eigen;
import org.jblas.Singular;
import org.jblas.Solve;
import org.junit.Test;

import darks.learning.dimreduce.pca.PCA;

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
        System.out.println(svd.getU());
        System.out.println(svd.getS());
        System.out.println(svd.getV());
		
		DoubleMatrix[] usv = Singular.fullSVD(new DoubleMatrix(testData));
        System.out.println(usv[0]);
        System.out.println(usv[1]);
        System.out.println(usv[2]);

        DoubleMatrix U = usv[0];
        DoubleMatrix S = usv[1];
        DoubleMatrix V = usv[2];
        DoubleMatrix mt = new DoubleMatrix(3, 4);
        for (int i = 0; i < S.length; i++)
        {
            mt.put(i, i, S.get(i));
        }
        System.out.println(mt.toString().replace(";", "\n"));
        DoubleMatrix src = U.mmul(mt).mmul(V.transpose());
        System.out.println(src.toString().replace(";", "\n"));
        mt = Solve.pinv(mt);
        System.out.println(mt.toString().replace(";", "\n"));
	}

	@Test
	public void testMatrix()
	{
		double[][] input = { 
				{1, 2},
				{3, 4},
			};
		DoubleMatrix im = new DoubleMatrix(input);
		System.out.println(im.transpose());
		
	}

	@Test
	public void testMatrixMul()
	{
		double[][] input = { 
				{1, 2, 3},
				{3, 4, 5}
			};
		double[][] weight = { 
				{1, 2, 3, 4},
				{1, 2, 3, 4},
				{1, 2, 3, 4}
			};
		DoubleMatrix im = new DoubleMatrix(input);
		DoubleMatrix wm = new DoubleMatrix(weight);
		//System.out.println(im.mul(wm));
		System.out.println(im.mmul(wm));
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
		System.out.println(im.rowSums());
		System.out.println(im.columnSums());
		System.out.println(Integer.parseInt(String.valueOf(Long.MAX_VALUE)));
	}

	@Test
	public void testEigen()
	{
		double[][] src = new double[][]{
				{-1, -1, 0, 2, 0},
				{-2, 0, 0, 1, 1}
		};
		DoubleMatrix mt = new DoubleMatrix(src);
		DoubleMatrix covariance = mt.mmul(mt.transpose()).div(mt.columns);
		ComplexDoubleMatrix eigVal = Eigen.eigenvalues(covariance);
		ComplexDoubleMatrix[] eigVector = Eigen.eigenvectors(covariance);
		System.out.println(eigVal);
		System.out.println(Arrays.toString(eigVector));
		ComplexDoubleMatrix cvec = eigVector[0];

		for (int i = 0; i < eigVal.length - 1; i++)
		{
			for (int j = 0; j < eigVal.length - i - 1; j++)
			{
				double j1 = eigVal.get(j).real();
				double j2 = eigVal.get(j + 1).real();
				if (j2 > j1)
				{
					cvec.swapColumns(j, j + 1);
					eigVal.swapRows(j, j + 1);
				}
			}
		}
		cvec = cvec.transpose();
		System.out.println(cvec);
		DoubleMatrix mt2 = cvec.getReal();
		mt2 = mt2.getRange(0, 1, 0, mt2.columns);
		System.out.println(mt2.mmul(mt));
		
		System.out.println(PCA.dimensionReduction(mt, 1));
	}
}
