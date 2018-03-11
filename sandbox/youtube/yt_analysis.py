import fetch_videos
from pandas import DataFrame
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from itertools import chain

nltk.download('movie_reviews')
nltk.download('punkt')

def word_featureset(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_featureset(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_featureset(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

trainfeats = negfeats + posfeats

classifier = NaiveBayesClassifier.train(trainfeats)

# Get comments from my channel
# comments = fetch_videos.get_comment_threads()

# comments_tokenized = fetch_videos.get_videos_sorted().apply(lambda x: [t.lower().encode('utf-8').strip(":,.!?") for t in x.split()])
# comments_sentiment = comments_tokenized.apply(lambda x: classifier.prob_classify(word_feats(x)).prob('pos') - classifier.prob_classify(word_feats(x)).prob('neg'))
# all = pd.read_json(comments)

# all['tokenized'] = all['text'].apply(lambda x: [t.lower().encode('utf-8').strip(":,.!?") for t in x.split()] )
# all['sentiment'] = all['tokenized'].apply(lambda x: classifier.prob_classify(word_feats(x)).prob('pos') - classifier.prob_classify(word_feats(x)).prob('neg') )
#
# videos = all.videoId.unique()
# all[all.videoId==videos[1]]


test_sentence = "This is the best video I have ever heard!"
test_sentence_tokenized = word_tokenize(test_sentence.lower())

# test_sent_features = word_tokenize(test_sentence.lower())
test_sent_features = {
    'love': False, 'deal': False, 'tired': False, 'feel': False, 'is': True, 'am': False, 'an': False,
    'sandwich': False, 'ca': False, 'best': True, '!': True, 'what': False, 'i': True, '.': False,
    'amazing': False, 'horrible': False, 'sworn': False, 'awesome': False, 'do': False, 'good': False,
    'very': False, 'boss': False, 'beers': False, 'not': False, 'with': False, 'he': False, 'enemy': False,
    'about': False, 'like': False, 'restaurant': False, 'this': True, 'of': False, 'work': False,
    "n't": False, 'these': False, 'stuff': False, 'place': False, 'my': False, 'view': False
}

# vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))



# print(classifier.classify(test_sent_features))
# print(comments)
print(word_tokenize(test_sentence))
print("\n")
# print(classifier.prob_classify("that"))
