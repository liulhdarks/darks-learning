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
package darks.learning.common.trietree;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import darks.learning.common.trietree.TrieNode.TrieNodeType;

public class TrieTree<T>
{
    
    private TrieNode<T> root;
    
    public TrieTree()
    {
        root = new TrieNode<T>(TrieNodeType.ROOT);
    }
    
    public void insertAndTail(String val, T obj)
    {
    	insert(val, obj);
    	StringBuilder buf = new StringBuilder(val);
    	insert(buf.reverse().toString(), obj);
    }
    
    public void insert(String val, T obj)
    {
        val = val.toLowerCase();
        int len = val.length();
        TrieNode<T> node = root;
        for (int i = 0; i < len; i++)
        {
            char ch = val.charAt(i);
            node = insertChar(node, ch);
        }
        if (node != null && node != root)
        {
            node.setStringTail(true);
            node.setObject(obj);
        }
    }
    
    private TrieNode<T> insertChar(TrieNode<T> parent, char ch)
    {
    	Map<Character, TrieNode<T>> nodes = parent.getNodes();
        TrieNode<T> node = nodes.get(ch);
        if (node == null)
        {
            node = new TrieNode<T>(TrieNodeType.LEAF);
            nodes.put(ch, node);
        }
        if (node.getTrieNodeType() == TrieNodeType.LEAF)
        {
            node.setValue(ch);
            node.setTrieNodeType(TrieNodeType.BRANCH);
        }
        if (node.getValue() != ch)
        {
            throw new RuntimeException("Fail to insert char." + node.getValue()
                + " which " + ch);
        }
        return node;
    }
    
    public boolean contain(String s)
    {
        TrieNode<T> node = root;
        int index = 0;
        while (index < s.length())
        {
            char ch = s.charAt(index++);
            node = node.getChildNode(ch);
            if (node == null 
                || node.getTrieNodeType() != TrieNodeType.BRANCH
                || node.getValue() != ch)
            {
                return false;
            }
        }
        
        if (node != null && node != root && node.isStringTail())
        {
            return true;
        }
        return false;
    }
    
    public boolean endsWith(String s)
    {
    	StringBuilder buf = new StringBuilder(s);
    	return startWith(buf.reverse().toString());
    }
    
    public boolean startWith(String s)
    {
        TrieNode<T> node = root;
        int index = 0;
        while (index < s.length())
        {
            char ch = s.charAt(index++);
            node = node.getChildNode(ch);
            if (node == null 
                || node.getTrieNodeType() != TrieNodeType.BRANCH
                || node.getValue() != ch)
            {
                return false;
            }
        }
        
        if (node != null)
        {
            return true;
        }
        return false;
    }
    
    public List<T> getEndsWith(String s)
    {
    	StringBuilder buf = new StringBuilder(s);
    	return getStartsWith(buf.reverse().toString());
    }
    
    public List<T> getStartsWith(String s)
    {
    	List<T> result = new LinkedList<T>();
        TrieNode<T> node = root;
        int index = 0;
        while (index < s.length())
        {
            char ch = s.charAt(index++);
            node = node.getChildNode(ch);
            if (node == null 
                || node.getTrieNodeType() != TrieNodeType.BRANCH
                || node.getValue() != ch)
            {
                return result;
            }
        }
        
        if (node != null)
        {
        	if (node.isStringTail())
        	{
        		result.add(node.getObject());
        	}
        	findTailString(result, node);
        }
        return result;
    }
    
    private void findTailString(List<T> result, TrieNode<T> parent)
    {
    	for (Entry<Character, TrieNode<T>> entry : parent.getNodes().entrySet())
    	{
    		TrieNode<T> node = entry.getValue();
    		if (node.isStringTail())
    		{
    			result.add(node.getObject());
    		}
    		else
    		{
    			findTailString(result, node);
    		}
    	}
    }
    
    public LinkedList<String> querySubString(String s)
    {
        LinkedList<String> list = new LinkedList<String>();
        
        return list;
    }
    
    public String toString()
    {
        return toStringNode(root);
    }
    
    private String toStringNode(TrieNode<T> parent)
    {
        if (parent == null)
        {
            return "";
        }
        StringBuffer buf = new StringBuffer();
        buf.append('[');
        buf.append(parent.getValue());
        if (isHasChildren(parent))
        {
            buf.append(',');
        }
        for (TrieNode<T> node : parent.getNodes().values())
        {
            buf.append(toStringNode(node));
        }
        buf.append(']');
        return buf.toString();
    }
    
    private boolean isHasChildren(TrieNode<T> parent)
    {
        boolean result = false;
        for (TrieNode<T> node : parent.getNodes().values())
        {
            if (node != null && node.getTrieNodeType() == TrieNodeType.BRANCH)
            {
                result = true;
                break;
            }
        }
        return result;
    }
}
