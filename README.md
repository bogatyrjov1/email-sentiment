**Decription**

The goal of this project is to explore machine learning techniques by determining email sentiment based on supervised learning.

**How to run**

1) Clone this repo
2) Make sure you have the `conf/conf_sandbox.yml` file
3) Run `script/server`

This will spin up a network of 2 docker containers ( db and app ) and run bash inside app container.
When inside app container, run python scripts in this repo separately.

Datasets in position:
- Enron ( 434k emails, uncategorized )
- Hillary Clinton ( 8k emails, uncategorized )
- Some tweets ( 1.5m tweets, categorized by sentiment polarity )
- IMDB movie reviews ( categorized )
