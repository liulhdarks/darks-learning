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
package darks.learning.topic.lsa;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.jblas.DoubleMatrix;

import darks.learning.common.basic.TfIdf;

public class LsaModel implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -6114717399052206074L;

    Map<Integer, String> sentenceColumnIndexs = new HashMap<Integer, String>();
    
    Map<String, Integer> wordsRowIndexs = new HashMap<String, Integer>();
    
    int K = 300;
    
    DoubleMatrix preMatrix = null;
    
    DoubleMatrix preNorm = null;
    
    DoubleMatrix inverseS = null;
    
    DoubleMatrix Uk = null;
    
    TfIdf tfidf = null;
    
    public LsaModel()
    {
    	
    }
    
    

	public LsaModel(Map<Integer, String> sentenceColumnIndexs, Map<String, Integer> wordsRowIndexs,
			int k, DoubleMatrix preMatrix, DoubleMatrix preNorm, DoubleMatrix inverseS,
			DoubleMatrix uk, TfIdf tfidf)
	{
		super();
		this.sentenceColumnIndexs = sentenceColumnIndexs;
		this.wordsRowIndexs = wordsRowIndexs;
		K = k;
		this.preMatrix = preMatrix;
		this.preNorm = preNorm;
		this.inverseS = inverseS;
		Uk = uk;
		this.tfidf = tfidf;
	}



	public Map<Integer, String> getSentenceColumnIndexs()
	{
		return sentenceColumnIndexs;
	}

	public void setSentenceColumnIndexs(Map<Integer, String> sentenceColumnIndexs)
	{
		this.sentenceColumnIndexs = sentenceColumnIndexs;
	}

	public Map<String, Integer> getWordsRowIndexs()
	{
		return wordsRowIndexs;
	}

	public void setWordsRowIndexs(Map<String, Integer> wordsRowIndexs)
	{
		this.wordsRowIndexs = wordsRowIndexs;
	}

	public int getK()
	{
		return K;
	}

	public void setK(int k)
	{
		K = k;
	}

	public DoubleMatrix getPreMatrix()
	{
		return preMatrix;
	}

	public void setPreMatrix(DoubleMatrix preMatrix)
	{
		this.preMatrix = preMatrix;
	}

	public DoubleMatrix getPreNorm()
	{
		return preNorm;
	}

	public void setPreNorm(DoubleMatrix preNorm)
	{
		this.preNorm = preNorm;
	}

	public DoubleMatrix getInverseS()
	{
		return inverseS;
	}

	public void setInverseS(DoubleMatrix inverseS)
	{
		this.inverseS = inverseS;
	}

	public DoubleMatrix getUk()
	{
		return Uk;
	}

	public void setUk(DoubleMatrix uk)
	{
		Uk = uk;
	}

	public TfIdf getTfidf()
	{
		return tfidf;
	}

	public void setTfidf(TfIdf tfidf)
	{
		this.tfidf = tfidf;
	}
    
    
}
