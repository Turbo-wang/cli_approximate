import re
import gensim
from collections import Counter


stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
stopwords_set = set(stopwords)
#statistic word frequency from sentences list
def count_word_fre(sents_list):
    sents = []
    for sen in sents_list:
        sents += sen
    word_fre = Counter(sents)
    return word_fre


# read raw file line by line.
def get_txt_line(file_name):
    # translator = str.maketrans({key: None for key in string.punctuation})
    with open(file_name) as f:
        for line in f:
            line = line.strip().lower()
            line = re.sub("[\"#$%&()*+-/:<=>@[\\]^_`{|}~]", " ", line)
            # line = line.translate(translator)
            tmp = re.split('\.|!|\?', line)
            for se in tmp:
                if len(se.split()) > 60:
                    subtmp = re.split(",|;", se)
                    for subse in subtmp:
                        sen = word_tokenize(se)
                        sen = list(set(sen) - stopwords_set)
                        yield sen
                else:
                    sen = word_tokenize(se)
                    sen = list(set(sen) - stopwords_set)
                    yield sen


def get_pku_ZN_line(file_name):
    #     chi_punc_list = ["。", "？", "！", "，", "、", "；", "：", "“", "”",\
    # "’", "‘", "（", "）", "[", "]", "{", "}", "【", "】", "——", "······", "···", "－", "～", "《》", "〈〉"]
    sentences = []
    with open(file_name) as f:
        for line in f:
            sentence = []
            for key in line.strip().split():
                sentence.append(key)
            sentences.append(sentence)
    return sentences


# get sentences list from raw file
def build_sent_list(file_name):
    sent_gene = get_txt_line(os.path.join(corpusDir, file_name))
    sent_list = []
    for sen in sent_gene:
        sent_list.append(sen)
    return sent_list


def process_corpus(sentence_list, fre_num, save_file):
    word_fre = count_word_fre(sentences_list)
    frequent_word_fre_tuple = word_fre.most_common(30000)
    frequent_word_fre_set = {}
    for tu in frequent_word_fre_tuple:
        frequent_word_fre_set.add(tu[0])
    sentence_pro_list = []
    for sentence in sentence_list:
        sentence_write = []
        for word in sentence:
            if word in frequent_word_fre_set:
                if word.isdigit():
                    sentence_write.append("DG")
                else:
                    sentence_write.append(word)
            else:
                if word.isdigit():
                    sentence_write.append("NUMBER")
                else:
                    sentence_write.append("UNKOWN_TOKEN")
        sentence_pro_list.append(sentence_write)
    if save_file:
        with open(os.path.join(corpusDir, save_file), 'w') as f:
            for sentence_pro in sentence_pro_list:
                f.write(" ".join(sentence_pro))
                f.write("\n")


def load_process_file(file_name):
    sentence_list = []
    with open(os.path.join(corpusDir, file_name)) as f:
        for line in f:
            sentences_list.append(line.strip().split())
    return sentence_list
