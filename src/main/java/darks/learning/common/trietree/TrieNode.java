package darks.learning.common.trietree;

import java.util.HashMap;
import java.util.Map;

public class TrieNode<T>
{
    private static final char CH_LEAF = '$';
    
    private static final char CH_ROOT = '#';
    
    public enum TrieNodeType
    {
        ROOT, BRANCH, LEAF
    };
    
    private char value;
    
    private TrieNodeType nodeType;
    
    private Map<Character, TrieNode<T>> nodes = new HashMap<Character, TrieNode<T>>();
    
    private boolean stringTail;
    
    T object;
    
    public TrieNode()
    {
        value = CH_LEAF;
        nodeType = TrieNodeType.LEAF;
        stringTail = false;
    }
    
    public TrieNode(TrieNodeType nodeType)
    {
        if (nodeType == TrieNodeType.ROOT)
        {
            value = CH_ROOT;
        }
        else if (nodeType == TrieNodeType.LEAF)
        {
            value = CH_LEAF;
        }
        this.nodeType = nodeType;
        stringTail = false;
    }
    
    public TrieNode(char value, T object, TrieNodeType nodeType)
    {
        this.nodeType = nodeType;
        this.value = value;
        this.object = object;
    }

    public TrieNodeType getTrieNodeType()
    {
        return nodeType;
    }

    public void setTrieNodeType(TrieNodeType nodeType)
    {
        this.nodeType = nodeType;
    }

    public char getValue()
    {
        return value;
    }

    public void setValue(char value)
    {
        this.value = value;
    }
    
    public TrieNode<T> getChildNode(char ch)
    {
    	return nodes.get(ch);
    }

    public boolean isStringTail()
    {
        return stringTail;
    }

    public void setStringTail(boolean stringTail)
    {
        this.stringTail = stringTail;
    }

    
	public Map<Character, TrieNode<T>> getNodes()
	{
		return nodes;
	}
	
	

	public T getObject()
	{
		return object;
	}

	public void setObject(T object)
	{
		this.object = object;
	}

	@Override
	public int hashCode()
	{
		final int prime = 31;
		int result = 1;
		result = prime * result + value;
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
		@SuppressWarnings("unchecked")
		TrieNode<T> other = (TrieNode<T>) obj;
		if (value != other.value)
			return false;
		return true;
	}
    
    
}
