# End-to-End Similarity Search

In this project I built a start to finish unsupervised image similarity search pipeline 
for any image corpus.  I will discuss the modules that connect together to make it happen. 
* A variational autoencoder for generating image similarity vectors
* A simple hashing technique for fast search response
* A database for storing the hashes and metadata
* A simple web UI for querying and viewing results

The technologies used are:
* Pytorch for the autoencoder
* MLFlow for dashboard tracking
* Faiss for vector search and storage
* Plotly-Dash for an interactive UI

First we will build a VAE (variational auto-encoder) to produce vector representations 
of input images that capture embeddings and respond to eucledean distance. 
(You can always use archtectures like VAEGAN to train custom distance metrics and/or
experiment with hamming, MSE or other effective distance measures!)  We then create a 
containerized service using the encoder portion of the nerual net.

Second, we will stand up a database and load it with the vectors from the training set. 

Lastly we will construct a simple UI with the ability to upload images and return 
the most similar entries.

### Download a dataset
I am going to use CIFAR-10:

```
wget http://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz
```

### Build the Auto-Encoder

### Stand Up the Database

### Load the Database

### Assemble the UI

### Query!
