# Flora text - topic modeling V0.1

## Data

   The intial dataset [enwiki-latest-pages-articles.xml.bz2](http://download.wikimedia.org/enwiki/)

   size : 14GB

   size after uncompressing : 34GB

## Getting Started

### preparing the data 

   ```shell

   wget "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

   python3 -m gensim.scripts.make_wiki enwiki-latest-pages-articles.xml.bz2 /PATH/TO/STORE
   ```

> this script takes about ~7 hours


### Training the model

   ```shell
   Python 3.5.2 (default, Sep 14 2017, 22:51:06) 
   Type 'copyright', 'credits' or 'license' for more information
   IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.

   In [1]: from topic_modeling.topic import TopicModeling
   In [2]: topic_modeling_class = TopicModeling()
   In [3]: topic_modeling_class.train()
   In [4]: exit
   ```
   > this script takes about 18 hours
