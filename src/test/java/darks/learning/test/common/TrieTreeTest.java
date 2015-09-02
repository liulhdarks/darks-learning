package darks.learning.test.common;

import org.junit.Test;

import darks.learning.common.trietree.TrieTree;

public class TrieTreeTest
{

	@Test
	public void testTrieTree()
	{
		TrieTree<String> tree = new TrieTree<String>();
		tree.insertAndTail("今", "今");
		tree.insertAndTail("今天", "今天");
		tree.insertAndTail("今天很", "今天很");
		tree.insertAndTail("今天不", "今天不");
		System.out.println(tree);
		System.out.println(tree.getStartsWith("今天"));
		System.out.println(tree.getEndsWith("很"));
		System.out.println(tree.getEndsWith("天"));
	}
	
}
