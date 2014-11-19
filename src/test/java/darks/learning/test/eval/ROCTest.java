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
package darks.learning.test.eval;

import org.junit.Test;

import darks.learning.eval.ROC;

public class ROCTest
{

	@Test
	public void testROC()
	{
		int lineId = 1;
		ROC roc = new ROC();
		roc.addLine(lineId, "test");
		roc.addPoint(lineId, 0.2, 0.1, "v1");
		roc.addPoint(lineId, 0.4, 0.4, "v2");
		roc.addPoint(lineId, 0.3, 0.6, "v3");
		roc.addPoint(lineId, 0.5, 0.7, "v4");
		roc.addPoint(lineId, 0.3, 0.2, "v5");
		roc.addPoint(lineId, 0.6, 0.8, "v6");
		roc.evalPlot(lineId, 1);
//		roc.showPlot();
		
        System.out.println("debug");
	}
	
}
