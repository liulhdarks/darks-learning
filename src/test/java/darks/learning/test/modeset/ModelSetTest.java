package darks.learning.test.modeset;

import org.junit.Test;

import darks.learning.model.ModelLoader;
import darks.learning.model.ModelSet;

public class ModelSetTest
{

	@Test
	public void testLoadMedel()
	{
		ModelSet modelSet = ModelLoader.loadFromStream(ModelSetTest.class.getResourceAsStream("/train_data.txt"));
		System.out.println(modelSet);
	}
	
}
