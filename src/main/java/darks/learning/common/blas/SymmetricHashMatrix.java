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
 * Symmetric Matrix with hash key
 * @author lihua.llh
 *
 * @param <T>
 */
public class SymmetricHashMatrix<T> extends Matrix<T>
{

	Map<Integer, T> valueMap = new HashMap<Integer, T>();

	int length;

	public SymmetricHashMatrix(int n)
	{
		length = (1 + n) * n / 2;
		rowsCount = n;
		columnsCount = n;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void put(int i, int j, T v)
	{
		if (i >= j)
			valueMap.put(i * (i + 1) / 2 + j, v);
		else
			valueMap.put(j * (j + 1) / 2 + i, v);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public T get(int i, int j)
	{
		if (i >= j)
			return valueMap.get(i * (i + 1) / 2 + j);
		else
			return valueMap.get(j * (j + 1) / 2 + i);
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

}
