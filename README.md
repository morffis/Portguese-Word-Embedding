# Portguese-Word-Embedding
The basics of word embedding with Portuguese language and using TensorFlow 2.0!

The model used in this code is the [GLOVE 50 dim](http://www.nilc.icmc.usp.br/embeddings)

This project enables to transform a text-based data set into a sequence of vectors to be used on machine learning projects.
I've reduced the size of the base csv "description_list" (the original is over 100k rows) just to show the base structed used.
The dataset consists of only two columns, [Description] and [Category].

This is a base framework that allow to any given text, we should be able to train a model to predict its category.
