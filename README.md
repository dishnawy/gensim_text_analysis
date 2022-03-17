# Gensim text analysis
> This is an example of a production text analysis engine built on top of gensim, spacy, scikit-learn with microservice(aka. RESTful API) using Flask and gunicorn/nginx.


## Features
That project includes the following features:
* Text summarization
* Keyword extraction
* Topic modeling
* Sentiment analysis
* Emotion analysis

## Usage example
We will use a very small data crawled by [Floralytics](http://www.floralytics.com/demo-page/) crawler aka. Miner, as an example. You can feed it with any JSON data from an API or stored.

### Input Example:
```sh
"platform": "youtube",
"length": "214",
"request_id": "cf6b8674-d870-4450-98cb-1af2c4b3ab5c",
"timestamp": 1521384822.771282,
"publisher": "Sea of Thieves",
"url": "http://cdn.embedly.com/widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%2FcWLcC4CkCjY%3Fwmode%3Dopaque%26feature%3Doembed&wmode=opaque&url=http%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DcWLcC4CkCjY&image=https%3A%2F%2Fi.ytimg.com%2Fvi%2FcWLcC4CkCjY%2Fhqdefault.jpg&key=6efca6e5ad9640f180f14146a0bc1392&type=text%2Fhtml&schema=youtube",
"likes": "1063",
"platform_id": "v=cWLcC4CkCjY",
"dislikes": "72",
"comments": [
{
"text": "I can't wait to play this!!!\ufeff",
"likes": "27",
"cid": "UgiR2IOh4Q3YdHgCoAEC",
"author": "Suzy Lu"
},
{
"text": "RIP Scalebound, I have high hopes for this game ;)\ufeff",
"likes": "63",
"cid": "UgjY-vuzH3vFungCoAEC",
"author": "Smaow"
},
{
"text": "Hopefully Microsoft is not going to cancel this one too\ufeff",
"likes": "66",
"cid": "UghcjPkskmaQG3gCoAEC",
"author": "Nicolas Cortés"
},
{
"text": "Just the idea of this game is one of the best things in years. Just make the PC version great.\ufeff",
"likes": "117",
"cid": "UggxluFkGN5gdHgCoAEC",
"author": "Boomabanga"
},
{
"text": "Awesome stuff Rare, keep it up! :)\ufeff",
"likes": "6",
"cid": "UgjhhkJ6QRBUWHgCoAEC",
"author": "The Crow's Nest"
},
{
"text": "Great video, I'm looking forward to play this game. I hope we soon know something as much of the ship customization as the upgrades, I have a lot of curiosity\ufeff",
"likes": "3",
"cid": "UgiDU9EEKkNo-3gCoAEC",
"author": "Sergiodelbetis"
},
{
"text": "This looks super epic guys. good job.\ufeff",
"likes": "29",
"cid": "Ugg6Cca4WdzlP3gCoAEC",
"author": "ZeroCarbThirty"
},
{
"text": "holy eye contact batman....\ufeff",
"likes": "21",
"cid": "Uggsgg1redVn0ngCoAEC",
"author": "Keith Stephens"
},
{
"text": "Too bad for Scalebound but this will be really fun\ufeff",
"likes": "1",
"cid": "UggkRJ083XrYIXgCoAEC",
"author": "FreshPrinceYuup"
},
{
"text": "HYPE! Hope you all have a great 2017 :)\ufeff",
"likes": "18",
"cid": "UgibXQvqXymWangCoAEC",
"author": "XboxNation"
}
],
"views": "121589",
"title": "Sea of Thieves Inn-side Story #10: Co-Op Gameplay"
```
The nlp engine will take the recieve the json here and a set of keys to process like:
```sh
nlp_result = nlp(json_object, [('comments', 'text'), 'title'])
```
then the  nlp_result will look like: 
```sh
{  
"sentiment": { 
"score": 0.493470,
    "label": "positive"
    },
"emotion": {
"sadness": 0.007838,
    "joy": 1,
    "fear": 0.081043,
    "disgust": 0.006973,
    "anger": 0.020815
},
"keywords":[
"eye contact batman",
"Awesome stuff",
"epic",
"best"
],
"topics":[
"technology,computing,google,game,consoles,nintendo
]
}
```

## General comments:
* [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) & [LSI](https://en.wikipedia.org/wiki/Latent_semantic_analysis) models are used for text analysis. Please click on each to get to know more.

## Possible Improvements
* POS tagging
* Text classification
* Entity tagging
* Support of several language ex (Finnish, French, Spanish, etc)
* Language autodetection
* Machine translation
* Humor and Sarcasm Detection


## Release History

* 0.1.0
    * First version
* 0.0.1
    * Work in progress

## Meta

Dish – engmeldishnawy@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.
