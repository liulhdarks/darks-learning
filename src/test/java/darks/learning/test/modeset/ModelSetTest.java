package darks.learning.test.modeset;

import java.io.File;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import darks.learning.corpus.Documents;
import darks.learning.corpus.Documents.Document;
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
	
	@Test
	public void testCorpus()
	{
		File input = new File("corpus/train_qianniu.txt");
        File labels = new File("corpus/train_qianniu_labels.txt");
        try
		{
            Documents docs = Documents.loadFromFile(input, labels, "UTF-8");
            Map<String, List<Document>> labelsMap = docs.getLabelsMap();
            Map<String, String> docsMap = docs.getDocsMap();
            System.out.println(labelsMap.size());
            System.out.println(docsMap.size());
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
}
