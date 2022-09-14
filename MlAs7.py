import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()

# single word lemmatization examples
list1 = ['classrooms', 'babies', 'dogs', 'puppies', 'smiling',
		'driving', 'days', 'tried', 'feet','children']
for words in list1:
	print(words + " ---> " + wnl.lemmatize(words))
	

# sentence lemmatization examples
string = 'the quick brown fox jumps over the lazy dog'

# Converting String into tokens
list2 = nltk.word_tokenize(string)
print(list2)

lemmatized_string = ' '.join([wnl.lemmatize(words) for words in list2])

print(lemmatized_string)
