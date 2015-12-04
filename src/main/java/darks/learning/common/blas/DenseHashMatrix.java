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

package darks.learning.common.blas;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Dense matrix with hash key
 * @author lihua.llh
 *
 * @param <T>
 */
public class DenseHashMatrix<T> extends Matrix<T>
{
	Map<Integer, T> valueMap = new HashMap<Integer, T>();
	
	public DenseHashMatrix(int n)
	{
		this(n, n);
	}
	
	public DenseHashMatrix(int n, int m)
	{
		rowsCount = n;
		columnsCount = m;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void put(int i, int j, T v)
	{
		valueMap.put(index(i, j), v);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public T get(int i, int j)
	{
		return (T)valueMap.get(index(i, j));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public boolean checkNull()
	{
		for (Entry<Integer, T> entry: valueMap.entrySet())
		{
			if (entry.getValue() != null)
				return false;
		}
		return true;
	}
	
	private int index(int i, int j)
	{
		return j * columnsCount + i;
	}

}
