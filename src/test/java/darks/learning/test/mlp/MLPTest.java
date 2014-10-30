package darks.learning.test.mlp;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import darks.learning.neuron.mlp.MultiLayerNeuronNetwork;

public class MLPTest
{
    
    @Test
    public void testMLP()
    {
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
    
        MultiLayerNeuronNetwork mlp = new MultiLayerNeuronNetwork();
        mlp.config.setHiddenLayouts(new int[]{10})
                    .setUseAdaGrad(true)
                    .setLearnRate(0.001)
                    .setMaxIterateCount(30000)
                    .setInputLayerSize(9)
                    .setOutputLayerSize(3)
                    .setUseRegularization(false);
        mlp.train(new DoubleMatrix(trainX), new DoubleMatrix(labels));
        
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
        
        DoubleMatrix testRet = mlp.predict(new DoubleMatrix(testX));
        System.out.println(testRet.toString().replace(";", "\n"));
    }
    
}
