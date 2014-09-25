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
package darks.learning.common.basic;

import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Haffman tree
 * @author Darks.Liu
 *
 */
public class Haffman
{

	private static Logger log = LoggerFactory.getLogger(Haffman.class);
	
	private TreeSet<HaffNode> elSet;
	
	private Collection<? extends HaffNode> elements;
	
	public Haffman(Collection<? extends HaffNode> elements)
	{
		elSet = new TreeSet<HaffNode>(elements);
		this.elements = elements;
	}
	
	public Haffman(Collection<HaffNode> elements, Comparator<HaffNode> comparator)
	{
		elSet = new TreeSet<HaffNode>(comparator);
		elSet.addAll(elements);
		this.elements = elements;
	}
	
	/**
	 * Build haffman tree
	 */
	public HaffNode build()
	{
		return build(0);
	}
	
	/**
	 * Build haffman tree
	 * 
	 * @param weightSize Weight size
	 */
	public HaffNode build(int weightSize)
	{
		while (elSet.size() > 1)
		{
			HaffNode node1 = elSet.pollFirst();
			HaffNode node2 = elSet.pollFirst();
			HaffNode parent = null;
			if (weightSize > 0)
			{
				parent = new WeightHaffNode(weightSize);
			}
			else
			{
				parent = new HaffNode();
			}
			parent.value = node1.value + node2.value;
			node1.parent = parent;
			node2.parent = parent;
			node1.code = 0;
			node2.code = 1;
			elSet.add(parent);
		}
		buildCodePath();
		return elSet.first();
	}
	
	private void buildCodePath()
	{
		log.debug("Building haffman code path for elements " + elements.size());
		for (HaffNode node : elements)
		{
			LinkedList<HaffNode> nodes = new LinkedList<HaffNode>();
			HaffNode curNode = node;
			while ((curNode = curNode.parent) != null)
			{
				nodes.addFirst(curNode);
			}
			node.codeNodes = nodes;
			int pathSize = nodes.size();
			if (pathSize == 0)
			{
				node.codePath = new int[1];
				node.codePath[0] = node.code;
			}
			else
			{
				node.codePath = new int[pathSize];
				for (int i = 1; i < pathSize; i++)
				{
					node.codePath[i - 1] = nodes.get(i).code;
				}
				node.codePath[pathSize - 1] = node.code;
			}
		}
	}
}
