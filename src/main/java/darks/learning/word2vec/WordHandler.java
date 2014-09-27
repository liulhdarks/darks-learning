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
package darks.learning.word2vec;

import java.util.List;
import java.util.Map;

public abstract class WordHandler
{

    protected Word2VecConfig config;
	
    protected double[] expTable;
    
    protected Map<String, WordNode> wordNodeMap;
	
    protected double learnRate;
    
    public WordHandler()
    {
    }
	
	public WordHandler(Word2Vec word3vec)
	{
		config = word3vec.getConfig();
		expTable = word3vec.getExpTable();
		learnRate = word3vec.getLearnRate();
		wordNodeMap = word3vec.getWordNodes();
	}
	
	/**
	 * Handle word2vec algorithm
	 * 
	 * @param index The word index of sentence
	 * @param sentence words set in sentence
	 * @param winScope Random window scope
	 */
	public abstract void handle(int index, List<WordNode> sentence);

	public double getLearnRate()
	{
		return learnRate;
	}

	public void setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
	}
	
}
