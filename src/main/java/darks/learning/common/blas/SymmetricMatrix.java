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

/**
 * Symmetric Matrix
 * @author lihua.llh
 *
 * @param <T>
 */
public class SymmetricMatrix<T> extends Matrix<T>
{

	Object[] values;

	int length;

	public SymmetricMatrix(int n)
	{
		length = (1 + n) * n / 2;
		values = new Object[length];
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
			values[i * (i + 1) / 2 + j] = v;
		else
			values[j * (j + 1) / 2 + i] = v;
	}

	/**
	 * {@inheritDoc}
	 */
	@SuppressWarnings("unchecked")
	@Override
	public T get(int i, int j)
	{
		if (i >= j)
			return (T)values[i * (i + 1) / 2 + j];
		else
			return (T)values[j * (j + 1) / 2 + i];
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public boolean checkNull()
	{
		for (Object obj : values)
		{
			if (obj != null)
				return false;
		}
		return true;
	}

}
