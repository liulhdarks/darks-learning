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
package darks.learning.dimreduce.plsa;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.rand.JdkRandomFunction;
import darks.learning.common.rand.RandomFunction;
import darks.learning.corpus.Corpus;

/**
 * Probabilistic Latent Semantic Analysis
 * @author Darks.Liu
 *
 */
public class ProbabilityLSA
{

    private static Logger log = LoggerFactory.getLogger(ProbabilityLSA.class);

    private int K = 300;
    
    RandomFunction randFunc = new JdkRandomFunction(System.currentTimeMillis());
    
    public ProbabilityLSA()
    {
    	
    }
    
    /**
     * Construction
     * @param targetDimension Target reduce dimension.Default 300
     */
    public ProbabilityLSA(int targetDimension)
    {
    	K = targetDimension > 0 ? targetDimension : K;
    }
    
    /**
     * Construction
     * @param targetDimension Target reduce dimension.Default 300
     * @param customRandFunc Custom random function.Default {@linkplain darks.learning.common.rand.JdkRandomFunction JdkRandomFunction}
     */
    public ProbabilityLSA(int targetDimension, RandomFunction customRandFunc)
    {
    	K = targetDimension > 0 ? targetDimension : K;
    	randFunc = customRandFunc != null ? customRandFunc : randFunc;
    }

    /**
     * Train model by corpus
     * @param corpus {@linkplain darks.learning.corpus.Corpus Corpus}
     */
    public void train(Corpus corpus)
    {
    	
    }
    
    
}
