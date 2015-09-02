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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

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

	private Map<K, Long> countMap = new ConcurrentHashMap<K, Long>();
	
	private long totalFreqCount = 0;
	
	public FreqCount()
	{
	}
	
	public void addValue(K key)
	{
		addValue(key, 1);
	}
	
	public synchronized void addValue(K key, int value)
	{
		Long count = countMap.get(key);
		if (count == null)
		{
			count = 0l;
		}
		totalFreqCount += value;
		countMap.put(key, count + value);
	}
	
	public synchronized void addValue(K key, long value)
	{
		Long count = countMap.get(key);
		if (count == null)
		{
			count = 0l;
		}
		totalFreqCount += value;
		countMap.put(key, count + value);
	}
	
	public synchronized Long getValue(K key)
	{
		Long count = countMap.get(key);
		if (count == null)
		{
			count = 0l;
			countMap.put(key, count);
		}
		return count;
	}
	
	public List<Entry<K, Long>> getSortList()
	{
		return getSortList(new Comparator<Map.Entry<K, Long>>()
        {
            @Override
            public int compare(Map.Entry<K, Long> o1, Map.Entry<K, Long> o2)
            {
                return o2.getValue().compareTo(o1.getValue());
            }
        });
	}
	
	public List<Entry<K, Long>> getSortList(Comparator<Map.Entry<K, Long>> comparator)
	{
		List<Map.Entry<K, Long>> sortIndexList = new ArrayList<Map.Entry<K, Long>>(countMap.entrySet());
        Collections.sort(sortIndexList, comparator);
        return sortIndexList;
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
	
	@Override
	public String toString()
	{
		StringBuilder buf = new StringBuilder();
		for (Entry<K, Long> entry : countMap.entrySet())
		{
			buf.append(entry.getKey()).append('\t').append(entry.getValue()).append('\n');
		}
		return buf.toString();
	}
	
	
}
