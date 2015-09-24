package darks.learning.common.basic;

import java.io.Serializable;

public class KeyValue<K, V> implements Serializable, Cloneable
{
	/**
	 * 
	 */
	private static final long serialVersionUID = -5105872297176893018L;

	K key;
	
	V value;

	public KeyValue()
	{
	}

	public KeyValue(K key, V value)
	{
		this.key = key;
		this.value = value;
	}

	public K getKey()
	{
		return key;
	}

	public void setKey(K key)
	{
		this.key = key;
	}

	public V getValue()
	{
		return value;
	}

	public void setValue(V value)
	{
		this.value = value;
	}

	@Override
	public String toString()
	{
		return "KeyValue [key=" + key + ", value=" + value + "]";
	}

	@Override
	public int hashCode()
	{
		final int prime = 31;
		int result = 1;
		result = prime * result + ((key == null) ? 0 : key.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj)
	{
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		KeyValue other = (KeyValue) obj;
		if (key == null)
		{
			if (other.key != null)
				return false;
		}
		else if (!key.equals(other.key))
			return false;
		return true;
	}
	
	
}
