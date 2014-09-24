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
package darks.learning.test.basic;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import darks.learning.common.basic.HaffNode;
import darks.learning.common.basic.Haffman;

public class HaffmanTest
{

	@Test
	public void testHaffman()
	{
		List<HaffNode> nodes = new ArrayList<HaffNode>();
		nodes.add(new HaffNode(1));
		nodes.add(new HaffNode(2));
		nodes.add(new HaffNode(4));
		nodes.add(new HaffNode(5));
		nodes.add(new HaffNode(7));
		Haffman haffman = new Haffman(nodes);
		HaffNode root = haffman.build(10);
		System.out.println(root);
	}
	
}
