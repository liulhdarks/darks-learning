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
package darks.learning.common.utils;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Calculate specify key's count
 * @author Darks.Liu
 *
 */
public class FreqCount<K> implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 6980004114842595199L;

	private Map<K, Long> countMap = new HashMap<K, Long>();
	
	private long totalFreqCount = 0;
	
	public FreqCount()
	{
	}
	
	public void addValue(K key)
	{
		addValue(key, 1);
	}
	
	public void addValue(K key, int value)
	{
		Long count = countMap.get(key);
		if (count == null)
		{
			count = 0l;
		}
		totalFreqCount += value;
		countMap.put(key, count + value);
	}
	
	public Long getValue(K key)
	{
		Long count = countMap.get(key);
		if (count == null)
		{
			count = 0l;
			countMap.put(key, count);
		}
		return count;
	}
	
	public Iterator<Entry<K, Long>> entrySetIterator()
	{
		return countMap.entrySet().iterator();
	}
	
	public long totalCount()
	{
		return totalFreqCount;
	}
	
	public long getUniqueCount()
	{
	    return countMap.size();
	}
}
