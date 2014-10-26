package darks.learning.test.math;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.jblas.DoubleMatrix;
import org.jblas.Singular;
import org.jblas.Solve;
import org.junit.Test;

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
//		double[][] output = { 
//				{1, 0, 0},
//				{0, 1, 0},
//				{0, 0, 1},
//			};
		DoubleMatrix im = new DoubleMatrix(input);
		DoubleMatrix wm = new DoubleMatrix(weight);
//		DoubleMatrix outm = new DoubleMatrix(output);
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
}
