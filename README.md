Darks Learning
==============

Darks learning is the machine learning algorithm library.
It contains Word2vec,DBN, RBM, MLP, LSA, PLSA, SDA, Maxent, regression, etc.

# Load Corpus
The corpus type are divided into Corpus, Documents and ModelSet.

## Corpus
It can be used for Word2vec,LSA,pLSA,etc. which are used to documents and words related to 
non classification algorithm. 
```Java
CorpusLoader loader = new CorpusLoader();
loader.addFilter(...);
loader.addStopwords(...);
Corpus corpus = loader.loadFromFile(new File(...));
```

## Documents
It can be used for Maxent, bayes, etc. which are used to documents and words related to 
classification algorithm.
```Java
File input = new File(...);
File labels = new File(...);
Documents docs = Documents.loadFromFile(input, labels, "UTF-8");
File corpusFile = new File(...); 
//The corpusFile both contains labels and features, which's labels and features of each line must be separated by Tab(\t).
Documents docs = Documents.loadFromFile(corpusFile, "UTF-8");
```

## ModelSet
It can be used for regression, MLP, DBN, RBM, SDA, SOFTMAX, etc. which are used to classification based on double matrix.
```Java
ModelSet modelSet = ModelLoader.loadFromFile(...);
ModelSet modelSet = ModelLoader.loadFromStream(...);
```

# Naive Bayes

Navie bayes contains the BINAMIAL and BERNOULLI modes. 
* BINAMIAL is fine-grained to words by default.
* BERNOULLI is coarse-grained to documents.

## Example
### How to train
```Java
Documents docs = Documents.loadFromFile(corpusFile, "UTF-8");
NaiveBayes bayes = new NaiveBayes();
bayes.config.setLogLikelihood(true)
			.setModelType(NaiveBayes.BINAMIAL);
bayes.train(docs);
```

### How to classify
```Java
String sentence = ...; //Sentence string must be separated by spaces.
String classify = bayes.predict(sentence);
String classify = bayes.predict(new String[]{...});
```

# Maxent

Maxent only supports GIS algorithm.

## Example
### How to train
```Java
Documents docs = Documents.loadFromFile(corpusFile, "UTF-8");
Maxent maxent = new GISMaxent();
MaxentModel model = maxent.train(docs, 1000); //Train model with 1000 iterations.
model.saveModel(...);
```

### How to load
```Java
GISModel model = GISModel.readModel(...);
GISMaxent maxent = new GISMaxent(model);
...
```

### How to classify
```Java
String[] terms = ...;
int labelIndex = maxent.predict(terms);
String classify = maxent.getLabel(labelIndex);
```
# LSA (Latent Semantic Analysis)
## Example
### How to train
```Java
CorpusLoader loader = new CorpusLoader(Corpus.TYPE_TF_IDF);
File file = ...;
Corpus corpus = loader.loadFromFile(file, "UTF-8");
LatentSemanticAnalysis lsa = new LatentSemanticAnalysis();
lsa.train(corpus);
lsa.saveModel(...);
lsa.loadModel(...);
```

### How to predict
```Java
String result = lsa.predict(new String[]{...});
int index = lsa.predictIndex(new String[]{...});
```



