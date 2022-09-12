from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language='english')
words = ['generous','generate','generously','generation']
for word in words:
  print(word,"--->",snowball.stem(word))
  print(SnowballStemmer.languages)
  from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language='german')
words = ['Es','sind' , 'sechzig', 'Sekunden', 'in', 'einer', 'Minute']
for word in words:
  print(word,"--->",snowball.stem(word))
  #import the nltk package
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
#create an object of class PorterStemmer
porter = PorterStemmer()
lancaster=LancasterStemmer()
snowball = SnowballStemmer('english')
#proide a word to be stemmed
print("Porter Stemmer")
print(porter.stem("cats"))
print(porter.stem("trouble"))
print(porter.stem("troubling"))
print(porter.stem("troubled"))
print("Lancaster Stemmer")
print(lancaster.stem("cats"))
print(lancaster.stem("trouble"))
print(lancaster.stem("troubling"))
print(lancaster.stem("troubled"))
print("Snowball Stemmer")
print(snowball.stem("cats"))
print(snowball.stem("trouble"))
print(snowball.stem("troubling"))
print(snowball.stem("troubled"))

# words to be stemmed
word_list = ["friend", "friendship", "friends", "friendships","stabil","destabilize","misunderstanding"]
print("{0:20}{1:20}{2:20}".format("Word","Porter Stemmer","lancaster Stemmer"))
for word in word_list:
  print("{0:20}{1:20}{2:20}".format(word,porter.stem(word),lancaster.stem(word)))
  file=open("review.txt")
my_lines_list=file.readlines()
my_lines_list

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
porter=PorterStemmer()
def stemSentence(sentence):
  token_words=word_tokenize(sentence)
  token_words
  stem_sentence=[]
  for word in token_words:
    stem_sentence.append(porter.stem(word))
    stem_sentence.append(" ")
    return "".join(stem_sentence)
print("Normal Review:")
print("")
print(my_lines_list[0])
print("")
print("Stemmed Review:")
print("")
x=stemSentence(my_lines_list[0])
print(x)
print("")
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.isri import ISRIStemmer
st = ISRIStemmer()
# file=open("arabic_review.txt")
w= "لقد أنهيت اللعبة للتو واستمتعت. كنت أقوم جزئيًا بتقييم اللعبة لابنتي التي هي من محبي القطط المتحمسين. كانت زوجتي قلقة بعض الشيء بشأن بعض الحالات التي يصاب فيها القط بجروح لفترة وجيزة ، لكنه عاد للوقوف على قدميه في وقت قصير. لذا بصرف النظر عن القليل من الشد على أوتار القلب بينما يعرج القط لمدة دقيقة أو دقيقتين ، فلا بأس. تحصل القصة على القليل من اللعبة المظلمة من منتصف إلى وقت متأخر. ربما تكون هناك بعض البيئات المخيفة التي تشق طريقك فيها. بالإضافة إل الخفيف واللعب الخفي. لكن لا شيء قد يكون في غير محله في لذلك أشعر أنه مناسب للأطفال. هناك بعض السيناريوهات التي يمكن أن تموت فيها القطة ، ولكن تتحول الشاشة إلى اللون الأحمر قبل إعادة التحميل السريع من الحفظ التلقائي. ليس هناك دماء أو صور مروعة. قد تكون طريقة اللعب"
for a in word_tokenize(w):
    print(st.stem(a))
    
    
    import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.isri import ISRIStemmer
st = ISRIStemmer()
# file=open("port_review.txt")
w= "Jogo de aventura distópico, estrelado por um gato. Acabei de terminar o jogo e me diver"
for a in word_tokenize(w):
 print(st.stem(a))

import nltk
nltk.download('rslp')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.rslp import RSLPStemmer
st = RSLPStemmer()
# file=open("port_review.txt")
w= "Jogo de aventura distópico, estrelado por um gato. Acabei de terminar o jogo e me diver"
for a in word_tokenize(w):
 print(st.stem(a))
