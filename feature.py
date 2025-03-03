# -*- coding = utf-8 -*-
# @Time : 2021/7/1 12:11
# @Author : ly
# @File : feature.py
# @Software: PyCharm
import os, re, sys
import numpy as np
import pandas as pd
from nltk.tree import Tree
from collections import Counter

current_dir = os.getcwd()  # obtain work dir
sys.path.append(current_dir)  # add work dir to sys path
import utils as us

def GetRoot():
    return os.path.abspath(os.path.join(os.getcwd(), ".."))


def GetResourceRoot():
    return os.path.join(os.getcwd(), 'resource')


def Getname(doc_path):
    return os.path.split(doc_path)[-1]



def RTXT(doc_path):
    docname,content=us.ReadDoc(doc_path)
    return docname

def TXT(doc_path):#存储
    data = {}
    data['name'] =['txtname']

    data[Getname(doc_path)] = RTXT(doc_path)
    TXT_df = pd.DataFrame(data)
    return TXT_df


def RStrokes(doc_path):#字形复杂度
    chars = us.LoadToken(doc_path, 'char')
    bihua_path = os.path.join(GetResourceRoot(), 'HZ2Bihua.txt')

    zi_bihua = us.CountRes_Dic(chars, bihua_path, 'Char', 'Stroke')
    argv_bh, maxi_bh, mini_bh = us.StatInfo(zi_bihua)
    low8_bh = len([i for i in zi_bihua if 0 < int(i) <= 8])
    hi16_bh = len([i for i in zi_bihua if int(i) >= 16])
    low10_bh = len([i for i in zi_bihua if 0 < int(i) <= 10])
    hi20_bh = len([i for i in zi_bihua if int(i) >= 20])

    corpus_path = os.path.join(GetResourceRoot(), 'Corpus_CH_List.txt')
    bh_freq_corpus = us.FreqWeight(zi_bihua, chars, corpus_path, 'Char', 'Frequency')

    subtitle_path = os.path.join(GetResourceRoot(), 'Subtitle_CH_List.txt')
    bh_freq_subtitle = us.FreqWeight(zi_bihua, chars, subtitle_path, 'Char', 'Frequency')

    stroke = [argv_bh, maxi_bh, mini_bh, low8_bh, hi16_bh, low10_bh, hi20_bh, bh_freq_corpus, bh_freq_subtitle]
    return stroke


def Strokes(doc_path):#存储
    data = {}
    data['name'] = 'argv_bh,maxi_bh,mini_bh,low8_bh,hi16_bh,low10_bh,hi20_bh,bh_freq_corpus,bh_freq_subtitle'.split(',')

    data[Getname(doc_path)] = RStrokes(doc_path)
    stroke_df = pd.DataFrame(data)
    return stroke_df


def RSym_once_wordChar(doc_path):
    chars = us.LoadToken(doc_path, 'char')
    words = us.LoadToken(doc_path, 'word')

    symm_path = os.path.join(GetResourceRoot(), 'SymmetryChar.txt')
    symm_char = len(us.CountRes_Lis(chars, symm_path))

    frequency_char = dict(Counter(chars))
    once_char = len([i for i in list(frequency_char.values()) if i == 1])

    frequency_word = dict(Counter(words))
    once_word = len([i for i in list(frequency_word.values()) if i == 1])

    charword_token = len(chars) / len(words) if len(words) != 0 else 0
    return [symm_char, once_char, once_word, charword_token]


def Sym_once(doc_path):
    data = {}
    data['name'] = 'symm_char,once_char,once_word,charword_token'.split(',')

    data[Getname(doc_path)] = RSym_once_wordChar(doc_path)
    sym_once_df = pd.DataFrame(data)
    return sym_once_df


def RDocFreq(doc_path):
    # print(doc_path)
    chars = us.LoadToken(doc_path, 'char')
    words = us.LoadToken(doc_path, 'word')
    # print(chars)
    # print(words)

    corpus_name = os.path.split(os.path.split(doc_path)[0])[-1]

    char_docfreq_path = os.path.join(GetResourceRoot(), '{}-chardocfreq.txt'.format(corpus_name))
    # print(char_docfreq_path)
    char_docfreq = us.CountRes_Dic(chars, char_docfreq_path, 'Char', 'Docfreq')#修改了大小写，原来为DocFreq
    # print(char_docfreq)
    argv_docfreq_char, maxi_docfreq_char, mini_docfreq_char = us.StatInfo(char_docfreq)
    var_docfreq_char = np.var(char_docfreq)

    word_docfreq_path = os.path.join(GetResourceRoot(), '{}-worddocfreq.txt'.format(corpus_name))
    word_docfreq = us.CountRes_Dic(chars, word_docfreq_path, 'Word', 'Docfreq')#修改了大小写，原来为DocFreq
    argv_docfreq_word, maxi_docfreq_word, mini_docfreq_word = us.StatInfo(word_docfreq)
    var_docfreq_word = np.var(word_docfreq)

    return argv_docfreq_char, maxi_docfreq_char, mini_docfreq_char, var_docfreq_char, argv_docfreq_word, maxi_docfreq_word, mini_docfreq_word, var_docfreq_word


def DocFreq(doc_path):
    data = {}
    data[
        'name'] = 'argv_docfreq_char,maxi_docfreq_char,mini_docfreq_char,var_docfreq_char,argv_docfreq_word,maxi_docfreq_word,mini_docfreq_word,var_docfreq_word'.split(
        ',')

    data[Getname(doc_path)] = RDocFreq(doc_path)
    docfreq_df = pd.DataFrame(data)
    return docfreq_df


def RCharFreq(doc_path):#汉字熟悉度
    corpus_path = os.path.join(GetResourceRoot(), 'Corpus_CH_List.txt')
    freqs_corpus_char = us.Freqencey(doc_path, 'char', corpus_path, 'Char', 'Frequency')
    subtitle_path = os.path.join(GetResourceRoot(), 'Subtitle_CH_List.txt')
    freqs_sub_char = us.Freqencey(doc_path, 'char', subtitle_path, 'Char', 'Frequency')

    rank_corpus_char = us.Freqencey(doc_path, 'char', corpus_path, 'Char', 'id')
    rank_sub_char = us.Freqencey(doc_path, 'char', subtitle_path, 'Char', 'id')

    re_freq = freqs_corpus_char + freqs_sub_char + rank_corpus_char + rank_sub_char
    return re_freq


def CharFreq(doc_path):
    data = {}
    data['name'] = """
    oov_corf_char,
    argv_corf_token_char,max_corf_token_char,
    argv_corf_type_char,max_corf_type_char,
    argv_logf_token_char,max_logf_token_char,
    argv_logf_type_char,max_logf_type_char,
    oov_subf_char,
    argv_subf_token_char,max_subf_token_char,
    argv_subf_type_char,max_subf_type_char,
    argv_sublogf_token_char,max_sublogf_token_char,
    argv_sublogf_type_char,max_sublogf_type_char,
    oov_corrank_char,
    argv_corrank_token_char,max_corrank_token_char,
    argv_corrank_type_char,max_corrank_type_char,
    argv_logrank_token_char,max_logrank_token_char,
    argv_logrank_type_char,max_logrank_type_char,
    oov_subrank_char,
    argv_subrank_token_char,max_subrank_token_char,
    argv_subrank_type_char,max_subrank_type_char,
    argv_sublogrank_token_char,max_sublogrank_token_char,
    argv_sublogrank_type_char,max_sublogrank_type_char
    """.replace('\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = RCharFreq(doc_path)
    # print(len(RCharFreq(doc_path)))
    # print(len(data['name']))
    charfreq_df = pd.DataFrame(data)
    return charfreq_df


def ROOV(doc_path):
    chars = us.LoadToken(doc_path, 'char')
    words = us.LoadToken(doc_path, 'word')

    corpus_char_path = os.path.join(GetResourceRoot(), 'Corpus_CH_List.txt')
    ResPath = corpus_char_path
    corf_char =us.LoadResource_Dic(ResPath, 'Char', 'Frequency').keys()#修改，增加ResPath
    #print(list(corf_char))转为list
    oov_corf500_char = len([i for i in chars if i not in list(corf_char)[:500]])#修改TypeError: 'dict_keys' object is not subscriptable，转为list
    oov_corf1000_char = len([i for i in chars if i not in list(corf_char)[:1000]])
    oov_corf1500_char = len([i for i in chars if i not in list(corf_char)[:1500]])
    oov_corf2000_char = len([i for i in chars if i not in list(corf_char)[:2000]])

    corpus_word_path = os.path.join(GetResourceRoot(), 'Corpus_Word_List.txt')
    ResPath = corpus_word_path#增加了这一步
    corf_word = us.LoadResource_Dic(ResPath, 'Word', 'Frequency').keys()

    oov_corf500_word = len([i for i in words if i not in list(corf_word)[:500]])#修改TypeError: 'dict_keys' object is not subscriptable，转为list
    oov_corf1000_word = len([i for i in words if i not in list(corf_word)[:1000]])
    oov_corf1500_word = len([i for i in words if i not in list(corf_word)[:1500]])
    oov_corf2000_word = len([i for i in words if i not in list(corf_word)[:2000]])

    commonchar_path = os.path.join(GetResourceRoot(), 'CommonUseChar.txt')
    common_char_35 = us.LoadResource_Lis(commonchar_path)
    oov_commonchar = len([i for i in chars if i not in common_char_35])

    return (oov_corf500_char, oov_corf1000_char, oov_corf1500_char, oov_corf2000_char, oov_commonchar,
            oov_corf500_word, oov_corf1000_word, oov_corf1500_word, oov_corf2000_word)


def OOV(doc_path):#词汇熟悉度
    data = {}
    data['name'] = """
    oov_corf500_char,oov_corf1000_char,oov_corf1500_char,oov_corf2000_char,oov_commonchar,
    oov_corf500_word,oov_corf1000_word,oov_corf1500_word,oov_corf2000_word""".replace('\n', '').replace(' ', '').split(
        ',')

    data[Getname(doc_path)] = ROOV(doc_path)
    # print(len(RCharFreq(doc_path)))
    # print(len(data['name']))
    oov_df = pd.DataFrame(data)
    return oov_df


def RDiffLevelChar(doc_path):
    HGJChar = os.path.join(GetResourceRoot(), 'HGJChar1.txt')
    newChar = os.path.join(GetResourceRoot(), 'NewHSKChar1.txt')
    oldChar = os.path.join(GetResourceRoot(), 'OldHSKChar1.txt')
    hgj_dl_char = us.CountLevel(doc_path, 'char', HGJChar, 'Char', 'Level')
    newhsk_dl_char = us.CountLevel(doc_path, 'char', newChar, 'Char', 'Level')
    oldhsk_dl_char = us.CountLevel(doc_path, 'char', oldChar, 'Char', 'Level')
    data = hgj_dl_char + newhsk_dl_char + oldhsk_dl_char
    return data


def DiffLevelChar(doc_path):#词语复杂度
    data = {}
    data['name'] = """
    argv_dl_tokenchar_hgj,var_dl_tokenchar_hgj,
    oov_tokenchar_hgj,dl01_tokenchar_hgj,dl02_tokenchar_hgj,dl03_tokenchar_hgj,dl04_tokenchar_hgj,
    oov_typechar_hgj,dl01_typechar_hgj,dl02_typechar_hgj,dl03_typechar_hgj,dl04_typechar_hgj,
    argv_dl_tokenchar_newhsk,var_dl_tokenchar_newhsk,
    oov_tokenchar_newhsk,dl01_tokenchar_newhsk,dl02_tokenchar_newhsk,dl03_tokenchar_newhsk,dl04_tokenchar_newhsk,dl05_tokenchar_newhsk,dl06_tokenchar_newhsk,
    oov_typechar_newhsk,dl01_typechar_newhsk,dl02_typechar_newhsk,dl03_typechar_newhsk,dl04_typechar_newhsk,dl05_typechar_newhsk,dl06_typechar_newhsk,
    argv_dl_tokenchar_oldhsk,var_dl_tokenchar_oldhsk,
    oov_tokenchar_oldhsk,dl01_tokenchar_oldhsk,dl02_tokenchar_oldhsk,dl03_tokenchar_oldhsk,dl04_tokenchar_oldhsk,
    oov_typechar_oldhsk,dl01_typechar_oldhsk,dl02_typechar_oldhsk,dl03_typechar_oldhsk,dl04_typechar_oldhsk
    """.replace('\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = RDiffLevelChar(doc_path)
    dlchar_df = pd.DataFrame(data)
    return dlchar_df


def RDiffLevelWord(doc_path):
    hgjword = os.path.join(GetResourceRoot(), 'HGJWord1.txt')
    huanyu = os.path.join(GetResourceRoot(), 'HuayuWordr1.txt')
    newWord = os.path.join(GetResourceRoot(), 'NewHSKWord1.txt')
    oldWord = os.path.join(GetResourceRoot(), 'OldHSKWord1.txt')

    hgj_dl_word = us.CountLevel(doc_path, 'word', hgjword, 'Word', 'Level')
    huayu_dl_word = us.CountLevel(doc_path, 'word', huanyu, 'Word', 'Level')
    newhsk_dl_word = us.CountLevel(doc_path, 'word', newWord, 'Word', 'Level')
    oldhsk_dl_word = us.CountLevel(doc_path, 'word', oldWord, 'Word', 'Level')
    return hgj_dl_word + huayu_dl_word + newhsk_dl_word + oldhsk_dl_word


def DiffLevelWord(doc_path):
    data = {}
    data['name'] = """argv_dl_tokenword_hgj,var_dl_tokenword_hgj,
    oov_tokenword_hgj,dl01_tokenword_hgj,dl02_tokenword_hgj,dl03_tokenword_hgj,dl04_tokenword_hgj,dl05_tokenword_hgj,dl06_tokenword_hgj,
    oov_typeword_hgj,dl01_typeword_hgj,dl02_typeword_hgj,dl03_typeword_hgj,dl04_typeword_hgj,dl05_typeword_hgj,dl06_typeword_hgj,
    argv_dl_tokenword_huayu,var_dl_tokenword_huayu,
    oov_tokenword_huayu,dl01_tokenword_huayu,dl02_tokenword_huayu,dl03_tokenword_huayu,dl04_tokenword_huayu,dl05_tokenword_huayu,dl06_tokenword_huayu,dl07_tokenword_huayu,
    oov_typeword_huayu,dl01_typeword_huayu,dl02_typeword_huayu,dl03_typeword_huayu,dl04_typeword_huayu,dl05_typeword_huayu,dl06_typeword_huayu,dl07_typeword_huayu,
    argv_dl_tokenword_newhsk,var_dl_tokenword_newhsk,
    oov_tokenword_newhsk,dl01_tokenword_newhsk,dl02_tokenword_newhsk,dl03_tokenword_newhsk,dl04_tokenword_newhsk,dl05_tokenword_newhsk,dl06_tokenword_newhsk,
    oov_typeword_newhsk,dl01_typeword_newhsk,dl02_typeword_newhsk,dl03_typeword_newhsk,dl04_typeword_newhsk,dl05_typeword_newhsk,dl06_typeword_newhsk,
    argv_dl_tokenword_oldhsk,var_dl_tokenword_oldhsk,
    oov_tokenword_oldhsk,dl01_tokenword_oldhsk,dl02_tokenword_oldhsk,dl03_tokenword_oldhsk,dl04_tokenword_oldhsk,
    oov_typeword_oldhsk,dl01_typeword_oldhsk,dl02_typeword_oldhsk,dl03_typeword_oldhsk,dl04_typeword_oldhsk""".replace(
        '\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = RDiffLevelWord(doc_path)
    dlword_df = pd.DataFrame(data)
    return dlword_df


def RDiffChar(doc_path):
    chars = us.LoadToken(doc_path, 'char')

    corpus_path = os.path.join(GetResourceRoot(), 'Corpus_CH_List.txt')
    subtitle_path = os.path.join(GetResourceRoot(), 'Subtitle_CH_List.txt')

    diffchartoken_1500_freq, diffchartype_1500_freq = us.CountDiff(chars, corpus_path, 'Char', 1500)
    diffchartoken_3000_freq, diffchartype_3000_freq = us.CountDiff(chars, corpus_path, 'Char', 3000)
    diffchartoken_1500_sub, diffchartype_1500_sub = us.CountDiff(chars, subtitle_path, 'Char', 1500)
    diffchartoken_3000_sub, diffchartype_3000_sub = us.CountDiff(chars, subtitle_path, 'Char', 3000)

    diffchar = [diffchartoken_1500_freq, diffchartype_1500_freq, diffchartoken_3000_freq, diffchartype_3000_freq,
                diffchartoken_1500_sub, diffchartype_1500_sub, diffchartoken_3000_sub, diffchartype_3000_sub]

    return diffchar


def DiffChar(doc_path):
    data = {}
    data['name'] = '''diffchartoken_1500_freq,diffchartype_1500_freq,diffchartoken_3000_freq,
    diffchartype_3000_freq,diffchartoken_1500_sub,diffchartype_1500_sub,diffchartoken_3000_sub,diffchartype_3000_sub'''.replace(
        '\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = RDiffChar(doc_path)
    diffchar_df = pd.DataFrame(data)
    return diffchar_df


def RDiffWord(doc_path):
    words = us.LoadToken(doc_path, 'word')

    corpus_path = os.path.join(GetResourceRoot(), 'Corpus_Word_List.txt')
    subtitle_path = os.path.join(GetResourceRoot(), 'Subtitle_Word_List.txt')
    modern_path = os.path.join(GetResourceRoot(), 'ContempWordf.txt')

    diffwordtoken_1500_freq, diffwordtype_1500_freq = us.CountDiff(words, corpus_path, 'Word', 1500)
    diffwordtoken_3000_freq, diffwordtype_3000_freq = us.CountDiff(words, corpus_path, 'Word', 3000)
    diffwordtoken_1500_sub, diffwordtype_1500_sub = us.CountDiff(words, subtitle_path, 'Word', 1500)
    diffwordtoken_3000_sub, diffwordtype_3000_sub = us.CountDiff(words, subtitle_path, 'Word', 3000)
    diffwordtoken_1500_contemp, diffwordtype_1500_contemp = us.CountDiff(words, modern_path, 'Word', 1500)
    diffwordtoken_3000_contemp, diffwordtype_3000_contemp = us.CountDiff(words, modern_path, 'Word', 3000)

    return (diffwordtoken_1500_freq, diffwordtype_1500_freq, diffwordtoken_3000_freq, diffwordtype_3000_freq,
            diffwordtoken_1500_sub, diffwordtype_1500_sub, diffwordtoken_3000_sub, diffwordtype_3000_sub,
            diffwordtoken_1500_contemp, diffwordtype_1500_contemp, diffwordtoken_3000_contemp,
            diffwordtype_3000_contemp)


def DiffWord(doc_path):
    data = {}
    data['name'] = """diffwordtoken_1500_freq,diffwordtype_1500_freq,diffwordtoken_3000_freq,diffwordtype_3000_freq,
        diffwordtoken_1500_sub,diffwordtype_1500_sub,diffwordtoken_3000_sub,diffwordtype_3000_sub,
        diffwordtoken_1500_contemp,diffwordtype_1500_contemp,diffwordtoken_3000_contemp,diffwordtype_3000_contemp""".replace(
        '\n', '').replace(' ', '').split(',')
    data[Getname(doc_path)] = RDiffWord(doc_path)
    diffword_df = pd.DataFrame(data)
    return diffword_df


def RCommonChar(doc_path):
    chars = us.LoadToken(doc_path, 'char')
    commonchar_path = os.path.join(GetResourceRoot(), 'CommonUseChar.txt')
    common_char_35 = us.LoadResource_Lis(commonchar_path)
    common_char_20 = common_char_35[:2001]
    common_char_10 = common_char_35[:1001]

    common_10 = len(us.CountUnion(chars, common_char_10))
    common_20 = len(us.CountUnion(chars, common_char_20))
    common_35 = len(us.CountUnion(chars, common_char_35))
    return [common_10, common_20, common_35]


def CommonChar(doc_path):
    data = {}
    data['name'] = 'common_10,common_20,common_35'.split(',')

    data[Getname(doc_path)] = RCommonChar(doc_path)
    common_df = pd.DataFrame(data)
    return common_df


def RTTR(doc_path):#ttr
    chars = us.LoadToken(doc_path, 'char')
    typo_char, ttr_char, root_char, log_char, mass_char, msttr_char = us.TTR(chars)

    words = us.LoadToken(doc_path, 'word')
    typo_word, ttr_word, root_word, log_word, mass_word, msttr_word = us.TTR(words)

    word_pos = us.LoadToken(doc_path, 'pos')
    content_tokens = us.ReturnContent(word_pos)
    typo_content, ttr_content, root_content, log_content, mass_content, msttr_content = us.TTR(content_tokens)

    return (typo_char, ttr_char, root_char, log_char, mass_char, msttr_char,
            typo_word, ttr_word, root_word, log_word, mass_word, msttr_word,
            typo_content, ttr_content, root_content, log_content, mass_content, msttr_content)


def TTR(doc_path):
    data = {}
    data['name'] = '''
    typo_char,ttr_char,root_char,log_char,mass_char,msttr_char,
    typo_word,ttr_word,root_word,log_word,mass_word,msttr_word,
    typo_content,ttr_content,root_content,log_content,mass_content,msttr_content'''.replace('\n', '').replace(' ',
                                                                                                              '').split(
        ',')
    data[Getname(doc_path)] = RTTR(doc_path)
    charttr_df = pd.DataFrame(data)
    return charttr_df


def RWordLen(doc_path):#词长
    words = us.LoadToken(doc_path, 'word')

    word_len = [len(i) for i in words]
    mean_wlen, maxi, mini = us.StatInfo(word_len)
    len1 = len([i for i in word_len if i == 1])
    len2 = len([i for i in word_len if i == 2])
    len3 = len([i for i in word_len if i == 3])
    len4 = len([i for i in word_len if i == 4])
    len4_ = len([i for i in word_len if i > 4])
    #叠词
    lenworddie4=len([i for i in [w for w in words if len(w)==4]if i[0]==i[1]])
    lenworddie4_ = len([i for i in [w for w in words if len(w) == 4] if i[2] == i[3]])
    lenworddie3 = len([i for i in [w for w in words if len(w) == 3] if i[0] == i[1]])
    lenworddie3_ = len([i for i in [w for w in words if len(w) == 3] if i[1] == i[2]])
    lenworddie2 = len([i for i in [w for w in words if len(w) == 2] if i[0] == i[1]])

    corpus_path = os.path.join(GetResourceRoot(), 'Corpus_Word_List.txt')
    wordlen_quali_cor = us.FreqWeight(word_len, words, corpus_path, 'Word', 'Frequency')

    sub_path = os.path.join(GetResourceRoot(), 'Subtitle_Word_List.txt')
    wordlen_quali_sub = us.FreqWeight(word_len, words, sub_path, 'Word', 'Frequency')

    wordlen_py = [us.CountWordPY(i) for i in words]
    wordlen_strokes = [us.CountWordStroke(i) for i in words]
    mean_wlen_py, maxi_py, mini_py = us.StatInfo(wordlen_py)
    mean_wlen_strokes, maxi_strokes, mini_strokes = us.StatInfo(wordlen_strokes)

    return mean_wlen, maxi, mini, len1, len2, len3, len4, len4_,lenworddie4,lenworddie4_,lenworddie3,lenworddie3_,lenworddie2, wordlen_quali_cor, wordlen_quali_sub, mean_wlen_py, maxi_py, mini_py, mean_wlen_strokes, maxi_strokes, mini_strokes


def WordLen(doc_path):
    data = {}
    data['name'] = '''mean_wlen,maxi,mini,len1,len2,len3,len4,len4_,lenworddie4,lenworddie4_,lenworddie3,lenworddie3_,lenworddie2,wordlen_quali_cor,wordlen_quali_sub,mean_wlen_py,maxi_py,mini_py,mean_wlen_strokes,maxi_strokes,mini_strokes'''.split(
        ',')

    data[Getname(doc_path)] = RWordLen(doc_path)
    charttr_df = pd.DataFrame(data)
    return charttr_df


def RWordFreq(doc_path):
    corpus_path = os.path.join(GetResourceRoot(), 'Corpus_Word_List.txt')
    freqs_corpus_word = us.Freqencey(doc_path, 'word', corpus_path, 'Word', 'Frequency')
    subtitle_path = os.path.join(GetResourceRoot(), 'Subtitle_Word_List.txt')
    freqs_sub_word = us.Freqencey(doc_path, 'word', subtitle_path, 'Word', 'Frequency')

    rank_corpus_word = us.Freqencey(doc_path, 'word', corpus_path, 'Word', 'id')
    rank_sub_word = us.Freqencey(doc_path, 'word', subtitle_path, 'Word', 'id')

    freqs_corpus_content = us.ContentFreq(doc_path, 'pos', corpus_path, 'Word', 'Frequency')
    freqs_sub_content = us.ContentFreq(doc_path, 'pos', subtitle_path, 'Word', 'Frequency')
    return freqs_corpus_word + freqs_sub_word + rank_corpus_word + rank_sub_word + freqs_corpus_content + freqs_sub_content


def WordFreq(doc_path):
    data = {}
    data['name'] = """oov_corf_word,
    argv_corf_token_word,max_corf_token_word,
    argv_corf_type_word,max_corf_type_word,
    argv_logf_token_word,max_logf_token_word,
    argv_logf_type_word,max_logf_type_word,
    oov_subf_word,
    argv_subf_token_word,max_subf_token_word,
    argv_subf_type_word,max_subf_type_word,
    argv_sublogf_token_word,max_sublogf_token_word,
    argv_sublogf_type_word,max_sublogf_type_word,
    oov_corrank_word,
    argv_corrank_token_word,max_corrank_token_word,
    argv_corrank_type_word,max_corrank_type_word,
    argv_logrank_token_word,max_logrank_token_word,
    argv_logrank_type_word,max_logrank_type_word,
    oov_subrank_word,
    argv_subrank_token_word,max_subrank_token_word,
    argv_subrank_type_word,max_subrank_type_word,
    argv_sublogrank_token_word,max_sublogrank_token_word,
    argv_sublogrank_type_word,max_sublogrank_type_word,
    argvfreqcor_content_tokens,argvfreqcor_content_types,
    argvfreqsub_content_tokens,argvfreqsub_content_types
    """.replace('\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = RWordFreq(doc_path)
    wordfreq_df = pd.DataFrame(data)
    return wordfreq_df


def NegWord(doc_path):
    '''
    内部
    '''
    words = us.LoadToken(doc_path, 'word')
    neg_path = os.path.join(GetResourceRoot(), 'Neg.txt')
    Negs = pd.read_csv(neg_path)
    Negs = Negs["NegWord"].tolist()
    neg = len([i for i in words for j in Negs if i.find(j) != -1])
    return neg


def Entity(doc_path):
    '''
    内部
    '''
    word_ner = us.LoadToken(doc_path, 'ner')
    entity = len([i.split('_')[0] for i in word_ner if i.find('O') == -1])
    return entity


def Concret(doc_path, concret_path):
    words = us.LoadToken(doc_path, 'word')
    Word_Conc = us.CountRes_Dic(words, concret_path, 'Word', 'Score')

    word_conc = [i for i in Word_Conc if i != 0]
    argv_conc, max_conc, min_conc = us.StatInfo(word_conc)
    var_conc = np.var(word_conc)
    return argv_conc, max_conc, min_conc, var_conc


def Cilin(doc_path, cilin_path):
    words = us.LoadToken(doc_path, 'word')
    Word_Symno = us.CountRes_Dic(words, cilin_path, 'Word', 'MultiMean')
    argv_symno, max_symno, min_symno = us.StatInfo(Word_Symno)
    var_symno = np.var(Word_Symno)
    return (argv_symno, max_symno, min_symno, var_symno)


def RWordOther(doc_path):
    '''
    外部调用
    '''
    entity = Entity(doc_path)
    neg = NegWord(doc_path)
    idiom_path = os.path.join(GetResourceRoot(), 'idioms.txt')
    idiom = us.CountSpecical(doc_path, 'word', idiom_path, "Idioms")

    yc_path = os.path.join(GetResourceRoot(), 'yc.txt')
    yc = us.CountSpecical(doc_path, 'word', yc_path, "词语")

    lianmian_path=os.path.join(GetResourceRoot(),'lianmian.txt')
    lianmian = us.CountSpecical(doc_path, 'word', lianmian_path, "词语")

    sensetouch_path=os.path.join(GetResourceRoot(), 'sense-touchwords.txt')
    sensetouchword = us.CountSpecical(doc_path, 'word', sensetouch_path, "词语")

    sensespace_path = os.path.join(GetResourceRoot(), 'sense-spacewords.txt')
    sensespaceword = us.CountSpecical(doc_path, 'word', sensespace_path, "词语")

    sensesight_path = os.path.join(GetResourceRoot(), 'sense-sightwords.txt')
    sensesightword = us.CountSpecical(doc_path, 'word', sensesight_path, "词语")

    sensehearing_path = os.path.join(GetResourceRoot(), 'sense-hearingwords.txt')
    sensehearingword = us.CountSpecical(doc_path, 'word', sensehearing_path, "词语")

    sensesmell_path = os.path.join(GetResourceRoot(), 'sense-smellwords.txt')
    sensesmellword = us.CountSpecical(doc_path, 'word', sensesmell_path, "词语")

    sensetaste_path = os.path.join(GetResourceRoot(), 'sense-tastewords.txt')
    sensetasteword = us.CountSpecical(doc_path, 'word', sensetaste_path, "词语")

    allsensewords_path = os.path.join(GetResourceRoot(), 'allsensewords.txt')
    allsensewords = us.CountSpecical(doc_path, 'word', allsensewords_path, "词语")

    dg_path = os.path.join(GetResourceRoot(), 'dg.xlsx')#典故词
    dg_word = us.CountSpecical_(doc_path, 'word', dg_path, "词语")

    gy_path = os.path.join(GetResourceRoot(), 'gywords.txt')#古语词
    gy_word = us.CountSpecical(doc_path, 'word', gy_path, "词语")

    sc_path = os.path.join(GetResourceRoot(), 'scwords.txt')#色彩词
    sc_word = us.CountSpecical(doc_path, 'word', sc_path, "词语")

    fy_path = os.path.join(GetResourceRoot(), 'fywords.txt')#方言词
    fy_word = us.CountSpecical(doc_path, 'word', fy_path, "词语")

    guanyongyu_path = os.path.join(GetResourceRoot(), 'guanyongyu.txt')
    guanyong_word = us.CountSpecical(doc_path, 'word', guanyongyu_path, "词语")

    suolueyu_path = os.path.join(GetResourceRoot(), 'suolueyu.txt')
    suolue_word = us.CountSpecical(doc_path, 'word', suolueyu_path, "词语")

    kouyu_path = os.path.join(GetResourceRoot(), 'kouyu.txt')
    kouyu_word = us.CountSpecical(doc_path, 'word', kouyu_path, "词语")

    cul_path = os.path.join(GetResourceRoot(), 'CulWords.txt')
    cul_word = us.CountSpecical(doc_path, 'word', cul_path, "CultureWords")

    meaning_path = os.path.join(GetResourceRoot(), 'WordMeaning.txt')
    argv_mean, max_mean, min_mean, var_mean = us.SpecicalWord_dic(doc_path, 'word', meaning_path, 'Word', 'Meaning')

    cul_path = os.path.join(GetResourceRoot(), 'WriteWord1.txt')
    argv_cul, max_cul, min_cul, var_cul = us.SpecicalWord_dic(doc_path, 'word', cul_path, 'Word', 'Level')

    mood_path = os.path.join(GetResourceRoot(), 'moodwords.txt')
    argv_mood_q, max_mood_q, min_mood_q, var_mood_q,sum_mood_q= us.CountSpecical__(doc_path, 'word', mood_path, '词语', '强度')
    argv_mood_j, max_mood_j, min_mood_j, var_mood_j ,sum_mood_j = us.CountSpecical__(doc_path, 'word', mood_path, '词语', '极性')

    highmeaning_path = os.path.join(GetResourceRoot(), 'high1.xlsx')
    highmeaning = us.CountSpecical_(doc_path, 'word', highmeaning_path, 'word')

    lowmeaning_path = os.path.join(GetResourceRoot(), 'low1.xlsx')
    lowmeaning = us.CountSpecical_(doc_path, 'word', lowmeaning_path, 'word')

    alleduwords_path = os.path.join(GetResourceRoot(), 'alleduwords.txt')
    argv_alleduwords_level, max_alleduwords_level, min_alleduwords_level, var_alleduwords_level = us.SpecicalWord_dic(doc_path, 'word', alleduwords_path, 'word', 'level')

    concret_path = os.path.join(GetResourceRoot(), 'Conc.txt')
    argv_conc, max_conc, min_conc, var_conc = Concret(doc_path, concret_path)

    cilin_path = os.path.join(GetResourceRoot(), 'Cilin.txt')
    argv_symno, max_symno, min_symno, var_symno = Cilin(doc_path, cilin_path)

    return (
    entity, neg, idiom, yc,lianmian,sensetouchword, sensespaceword, sensesightword, sensehearingword, sensesmellword, sensetasteword,
    allsensewords,dg_word, gy_word, sc_word, fy_word, guanyong_word, suolue_word, kouyu_word,cul_word, argv_mean, max_mean, min_mean, argv_cul,
    max_cul, min_cul, var_cul,argv_mood_q, max_mood_q, min_mood_q, var_mood_q,sum_mood_q,argv_mood_j, max_mood_j, min_mood_j, var_mood_j,sum_mood_j,
    highmeaning,
    lowmeaning,
    argv_alleduwords_level, max_alleduwords_level, min_alleduwords_level, var_alleduwords_level,
    argv_conc, max_conc, min_conc, var_conc,
    argv_symno, max_symno, min_symno, var_symno)


def WordOther(doc_path):
    data = {}
    data['name'] = ''' entity, neg, idiom, yc,lianmian,sensetouchword, sensespaceword, sensesightword, sensehearingword, sensesmellword, sensetasteword,
    allsensewords,dg_word, gy_word, sc_word, fy_word, guanyong_word, suolue_word, kouyu_word,cul_word, argv_mean, max_mean, min_mean, argv_cul,
    max_cul, min_cul, var_cul,argv_mood_q, max_mood_q, min_mood_q, var_mood_q,sum_mood_q,argv_mood_j, max_mood_j, min_mood_j, var_mood_j,sum_mood_j,
    highmeaning,
    lowmeaning,
    argv_alleduwords_level, max_alleduwords_level, min_alleduwords_level, var_alleduwords_level,
    argv_conc, max_conc, min_conc, var_conc,
    argv_symno, max_symno, min_symno, var_symno'''.replace('\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = RWordOther(doc_path)
    wordsother_df = pd.DataFrame(data)
    return wordsother_df

def Rwencaidic(doc_path):
    # tokens = us.LoadToken(doc_path, 'word')
    # highmeaning_path = os.path.join(GetResourceRoot(), 'high1.xlsx')
    # highmeaning = us.CountSpecical_(doc_path, 'word', highmeaning_path, 'word')
    # highmeaning_rate=highmeaning/len(tokens)
    # lowmeaning_path = os.path.join(GetResourceRoot(), 'low1.xlsx')
    # lowmeaning = us.CountSpecical_(doc_path, 'word', lowmeaning_path, 'word')
    # lowmeaning_rate=lowmeaning/len(tokens)
    wencaidic_path = os.path.join(GetResourceRoot(), 'wencai1.txt')
    argv_wencai_score, max_wencai_score, min_wencai_score, var_wencai_score ,sum_wencai_score= us. CountSpecical__(
        doc_path, 'word', wencaidic_path, 'Word', 'score')
    return (argv_wencai_score, max_wencai_score, min_wencai_score, var_wencai_score,sum_wencai_score)
    # return (highmeaning,highmeaning_rate,
    #         lowmeaning,lowmeaning_rate )

def wencaidic(doc_path):
    data = {}
    data['name'] = '''argv_wencai_score, max_wencai_score, min_wencai_score, var_wencai_score,sum_wencai_score'''.replace('\n', '').replace(' ', '').split(',')
    # data[
    #     'name'] = '''highmeaning,highmeaning_rate,
    #         lowmeaning,lowmeaning_rate'''.replace(
    #     '\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = Rwencaidic(doc_path)
    wencaidic_df = pd.DataFrame(data)
    return wencaidic_df

def CountMeanClass(sent, cilin_path):
    tokens = sent.split(' ')
    with open(cilin_path, encoding='utf-8') as f:
        cilin = f.readlines()
        cilin = [i.strip() for i in cilin]
        ci = [i.split(' ')[0] for i in cilin]
        classes = [i.split(' ')[1:] for i in cilin]
        ci_dic = dict(zip(ci, classes))
    ci_class = [ci_dic[t] for t in tokens if t in ci_dic.keys()]
    ci_class0 = len(set([j[0] for i in ci_class for j in i]))
    ci_class1 = len(set([j[:2] for i in ci_class for j in i]))
    ci_class2 = len(set([j[:3] for i in ci_class for j in i]))

    return ci_class0, ci_class1, ci_class2

def Countsynmean(tokens, cilin_path):
    with open(cilin_path, encoding='utf-8') as f:
        cilin = f.readlines()
        cilin = [i.strip() for i in cilin]
        ci = [i.split(' ')[0] for i in cilin]
        classes = [i.split(' ')[1:] for i in cilin]
        ci_dic = dict(zip(ci, classes))
        # print(ci_dic)
    ci_class = [ci_dic[t] for t in tokens if t in ci_dic.keys()]
    # print(ci_class)
    ci_classsyn1 = [j[:-1] for i in ci_class for j in i if j[-1]=='=']
    classsyn1 = {}
    for i in ci_classsyn1:
        if ci_classsyn1.count(i) > 1:
            classsyn1[i] = ci_classsyn1.count(i)
    ci_classsyn1=len(classsyn1)
    ci_classsyn2 = [j[:-1] for i in ci_class for j in i if j[-1]=='#']
    classsyn2 = {}
    for i in ci_classsyn2:
        if ci_classsyn2.count(i) > 1:
            classsyn2[i] = ci_classsyn2.count(i)
    ci_classsyn2 = len(classsyn2)
    return ci_classsyn1, ci_classsyn2

def Rsynmeaning(doc_path):
    tokens = us.LoadToken(doc_path, 'word')
    cilin_path = os.path.join(GetResourceRoot(), 'TongYiCL.txt')
    ci_classsyn1, ci_classsyn2=Countsynmean(tokens,cilin_path)

    return ci_classsyn1, ci_classsyn2

def synmeaning(doc_path):
    data = {}
    data['name'] = ''' ci_classsyn1, ci_classsyn2'''.replace('\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = Rsynmeaning(doc_path)
    synmeaning_df = pd.DataFrame(data)
    return synmeaning_df

def CountMeanwc(tokens, cilin_path):

    with open(cilin_path, encoding='utf-8') as f:
        cilin = f.readlines()
        cilin = [i.strip() for i in cilin]
        ci = [i.split(' ')[0] for i in cilin]
        classes = [i.split(' ')[1:] for i in cilin]
        ci_dic = dict(zip(ci, classes))
        # print(ci_dic)
    ci_class = [ci_dic[t] for t in tokens if t in ci_dic.keys()]
    # print(ci_class)
    # print(' '.join(j[:2] for i in ci_class for j in i))
    # print(' '.join(j[0] for i in ci_class for j in i))
    meaningPoses1 = ['A','B','C','D','E','F','G','H','I','J','K','L']
    meaningPoses2=['Aa','Ab','Ac','Ad','Ae','Af','Ag','Ah','Ai','Aj','Ak','Al','Am','An','Ba','Bb','Bc','Bd','Be','Bf','Bg','Bh','Bi','Bj','Bk','Bl','Bm','Bn','Bo','Bp','Bq','Br','Ca','Cb','Da','Db','Dc','Dd','De','Df','Dg','Dh','Di','Dj','Dk','Dl','Dm','Dn','Ea','Eb','Ec','Ed','Ee','Ef','Fa','Fb','Fc','Fd','Ga','Gb','Gc','Ha','Hb','Hc','Hd','He','Hf','Hg','Hh','Hi','Hj','Hk','Hl','Hm','Hn','Ia','Ib','Ic','Id','Ie','If','Ig','Ih','Ja','Jb','Jc','Jd','Je','Ka','Kb','Kc','Kd','Ke','Kf','La']
    A,B,C,D,E,F,G,H,I,J,K,L=[len([j[0] for i in ci_class for j in i if j[0]==wcpos]) for wcpos in meaningPoses1]
    Aa, Ab, Ac, Ad, Ae, Af, Ag, Ah, Ai, Aj, Ak, Al, Am, An, Ba, Bb, Bc, Bd, Be, Bf, Bg, Bh, Bi, Bj, Bk, Bl, Bm, Bn, Bo, Bp, Bq, Br, Ca, Cb, Da, Db, Dc, Dd, De, Df, Dg, Dh, Di, Dj, Dk, Dl, Dm, Dn, Ea, Eb, Ec, Ed, Ee, Ef, Fa, Fb, Fc, Fd, Ga, Gb, Gc, Ha, Hb, Hc, Hd, He, Hf, Hg, Hh, Hi, Hj, Hk, Hl, Hm, Hn, Ia, Ib, Ic, Id, Ie, If, Ig, Ih, Ja, Jb, Jc, Jd, Je, Ka, Kb, Kc, Kd, Ke, Kf, La=[len([j[:2] for i in ci_class for j in i if j[:2]==wcpos2]) for wcpos2 in meaningPoses2]

    return A,B,C,D,E,F,G,H,I,J,K,L,Aa, Ab, Ac, Ad, Ae, Af, Ag, Ah, Ai, Aj, Ak, Al, Am, An, Ba, Bb, Bc, Bd, Be, Bf, Bg, Bh, Bi, Bj, Bk, Bl, Bm, Bn, Bo, Bp, Bq, Br, Ca, Cb, Da, Db, Dc, Dd, De, Df, Dg, Dh, Di, Dj, Dk, Dl, Dm, Dn, Ea, Eb, Ec, Ed, Ee, Ef, Fa, Fb, Fc, Fd, Ga, Gb, Gc, Ha, Hb, Hc, Hd, He, Hf, Hg, Hh, Hi, Hj, Hk, Hl, Hm, Hn, Ia, Ib, Ic, Id, Ie, If, Ig, Ih, Ja, Jb, Jc, Jd, Je, Ka, Kb, Kc, Kd, Ke, Kf, La

def Rwcmeaning(doc_path):
    tokens = us.LoadToken(doc_path, 'word')
    cilin_path = os.path.join(GetResourceRoot(), 'TongYiCL.txt')
    A, B, C, D, E, F, G, H, I, J, K, L, Aa, Ab, Ac, Ad, Ae, Af, Ag, Ah, Ai, Aj, Ak, Al, Am, An, Ba, Bb, Bc, Bd, Be, Bf, Bg, Bh, Bi, Bj, Bk, Bl, Bm, Bn, Bo, Bp, Bq, Br, Ca, Cb, Da, Db, Dc, Dd, De, Df, Dg, Dh, Di, Dj, Dk, Dl, Dm, Dn, Ea, Eb, Ec, Ed, Ee, Ef, Fa, Fb, Fc, Fd, Ga, Gb, Gc, Ha, Hb, Hc, Hd, He, Hf, Hg, Hh, Hi, Hj, Hk, Hl, Hm, Hn, Ia, Ib, Ic, Id, Ie, If, Ig, Ih, Ja, Jb, Jc, Jd, Je, Ka, Kb, Kc, Kd, Ke, Kf, La =CountMeanwc(tokens,cilin_path)
    return A,B,C,D,E,F,G,H,I,J,K,L,Aa, Ab, Ac, Ad, Ae, Af, Ag, Ah, Ai, Aj, Ak, Al, Am, An, Ba, Bb, Bc, Bd, Be, Bf, Bg, Bh, Bi, Bj, Bk, Bl, Bm, Bn, Bo, Bp, Bq, Br, Ca, Cb, Da, Db, Dc, Dd, De, Df, Dg, Dh, Di, Dj, Dk, Dl, Dm, Dn, Ea, Eb, Ec, Ed, Ee, Ef, Fa, Fb, Fc, Fd, Ga, Gb, Gc, Ha, Hb, Hc, Hd, He, Hf, Hg, Hh, Hi, Hj, Hk, Hl, Hm, Hn, Ia, Ib, Ic, Id, Ie, If, Ig, Ih, Ja, Jb, Jc, Jd, Je, Ka, Kb, Kc, Kd, Ke, Kf, La

def wcmeaning(doc_path):
    data = {}
    data['name'] = '''A,B,C,D,E,F,G,H,I,J,K,L,
    Aa, Ab, Ac, Ad, Ae, Af, Ag, Ah, Ai, Aj, Ak, Al, Am, An, Ba, Bb, Bc, Bd, Be, Bf, Bg, Bh, Bi, Bj, Bk, Bl, Bm, Bn, Bo, Bp, Bq, Br, Ca, Cb, Da, Db, Dc, Dd, De, Df, Dg, Dh, Di, Dj, Dk, Dl, Dm, Dn, Ea, Eb, Ec, Ed, Ee, Ef, Fa, Fb, Fc, Fd, Ga, Gb, Gc, Ha, Hb, Hc, Hd, He, Hf, Hg, Hh, Hi, Hj, Hk, Hl, Hm, Hn, Ia, Ib, Ic, Id, Ie, If, Ig, Ih, Ja, Jb, Jc, Jd, Je, Ka, Kb, Kc, Kd, Ke, Kf, La'''.replace(
        '\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = Rwcmeaning(doc_path)
    wcmeaning_df = pd.DataFrame(data)
    return wcmeaning_df

def RSentComplex(doc_path):
    '''
    外部调用
    '''
    sents = us.LoadToken(doc_path, 'sent')

    easy_sent, complex_sent = us.EasyComplex(sents)
    mean_lenz, max_lenz, min_lenz, mean_lenw, max_lenw, min_lenw = us.CountLen(sents)
    mean_posnum, max_posnum, min_posnum = us.CountSentPos(doc_path)
    sf=[',',';',':','，','；','、','：']

    cilin_path = os.path.join(GetResourceRoot(), 'TongYiCL.txt')
    panc=''
    class0 = []
    class1 = []
    class2 = []
    for sent in sents:
        # print(sent)
        panc=panc+''.join(i for i in sent if i in sf)
        ci_class0, ci_class1, ci_class2 = CountMeanClass(sent, cilin_path)
        class0.append(ci_class0)
        class1.append(ci_class1)
        class2.append(ci_class2)
    argv_class0, max_class0, min_class0 = us.StatInfo(class0)
    argv_class1, max_class1, min_class1 = us.StatInfo(class1)
    argv_class2, max_class2, min_class2 = us.StatInfo(class2)
    sent = len(sents)
    SF=len(sf)/sent
    # print(SF,sent)

    return (SF,sent, easy_sent, complex_sent, mean_lenz, max_lenz, min_lenz, mean_lenw, max_lenw, min_lenw, mean_posnum,
            max_posnum, min_posnum,
            argv_class0, max_class0, min_class0, argv_class1, max_class1, min_class1, argv_class2, max_class2,
            min_class2)


def SentComplex(doc_path):
    data = {}
    data['name'] = '''SF,sent,easy_sent,complex_sent,mean_lenz,max_lenz,min_lenz,mean_lenw,max_lenw,min_lenw,mean_posnum,max_posnum,min_posnum,
        argv_class0,max_class0,min_class0,argv_class1,max_class1,min_class1,argv_class2,max_class2,min_class2'''.replace(
        '\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = RSentComplex(doc_path)
    sentcom_df = pd.DataFrame(data)
    return sentcom_df


def RWordPos(doc_path):
    '''
    外部调用
    '''
    word_pos = us.LoadToken(doc_path, 'pos')

    ContentPos = ['a', 'b', 'i', 'j', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'q', 'r',
                  'v']  # 实词：名词 动词 形容词 数词量词代词
    FuncPos = ['d', 'p', 'c', 'u', 'e', 'o']  # 副词、介词、连词、助词、语气词、叹词
    MOD=['a', 'b','d','z']
    poses = [ContentPos, FuncPos,MOD, 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl',
             'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z']
    cont, func,MOD, a, b, c, d, e, g, h, i, j, k, m, n, nd, nh, ni, nl, ns, nt, nz, o, p, q, r, u, v, wp, ws, x, z = [
        us.CountPos(word_pos, i) for i in poses]
    MVR=MOD/v
    # print(MVR)

    ranks = list(
        np.argsort([a, b, c, d, e, g, h, i, j, k, m, n, nd, nh, ni, nl, ns, nt, nz, o, p, q, r, u, v, wp, ws, x, z]))

    words = us.LoadToken(doc_path, 'word')
    '''
    PersonProuns = ['你','我','他','它','她','我们','他们','它们','她们']
    OneProuns = ['我','我们','自己']
    ThirdProuns = ['他','它','她','他们','它们','她们']
    Binglielis = ['和','跟','与','同','及','而','况','况且','何况','乃至','以及','同时','此外','还有','另外','既','且']  
    xuanzelis = ['或','还是','或者','与其','宁可','要么','亦或']  
    chengjielis = ['则','乃','就','而','于是','至于','说到','此外','像','如','一般','比方']  
    dijinlis = ['不但','不仅','而且','并且','况且','并','且','甚至','非但','尚且','何况','反而','相反']
    zhuangzhelis = ['却','但是','然而','而','偏偏','只是','不过','至于','致','不料','岂知','可是','只不过']
    yinguolis = ['原来','因为','由于','以便','因此','所以','是故','以致','既然','由于','于是','从而','那么']
    jiashelis = ['若','如果','若是','假如','假使','倘若','要是','譬如','万一','要不是','要不然','否则']
    bijiaolis = ['像','好比','如同','似乎','等于','不如','不及','与其','虽然','可是']
    rangbulis = ['虽然','固然','尽管','纵然','即使','哪怕','即','即便','然而']
    mudilis = ['以','以便','以免','为了','以免','以防','省得','免得','以求','借以']
    '''

    Proun_path = os.path.join(GetResourceRoot(), 'Pronoun.txt')
    Conj_path = os.path.join(GetResourceRoot(), 'Conj.txt')

    PronAll, PronDemo, PronPers, PronPers_Firs, PronPers_Secd, PronPers_Thir, PronWhat = us.LoadWordlis(Proun_path)
    binlieConj, dijinConj, jissheConj, rangbuConj, shunchengConj, tiaojianConj, yinguoConj, zhuanzheConj, ConjNeg, ConjPosi, ConjAll = us.LoadWordlis(
        Conj_path)

    others = [len(us.CountUnion(i, words)) for i in
              [PronAll, PronDemo, PronPers, PronPers_Firs, PronPers_Secd, PronPers_Thir, PronWhat, binlieConj,
               dijinConj, jissheConj, rangbuConj, shunchengConj, tiaojianConj, yinguoConj, zhuanzheConj, ConjNeg,
               ConjPosi, ConjAll]]

    wordpos = [cont, func,MOD, a, b, c, d, e, g, h, i, j, k, m, n, nd, nh, ni, nl, ns, nt, nz, o, p, q, r, u, v, wp, ws, x,
               z] +[MVR]+ ranks + others

    return wordpos


def WordPos(doc_path):
    data = {}
    data['name'] = '''cont,func,MOD,a,b,c,d,e,g,h,i,j,k,m,n,nd,nh,ni,nl,ns,nt,nz,o,p,q,r,u,v,wp,ws,x,z,MVR,
    a_rank,b_rank,c_rank,d_rank,e_rank,g_rank,h_rank,i_rank,j_rank,k_rank,m_rank,n_rank,nd_rank,nh_rank,ni_rank,nl_rank,ns_rank,nt_rank,nz_rank,o_rank,p_rank,q_rank,r_rank,u_rank,v_rank,wp_rank,ws_rank,x_rank,z_rank,
    PronAll,PronDemo,PronPers,PronPers_Firs,PronPers_Secd,PronPers_Thir,PronWhat,
    binlieConj,dijinConj,jissheConj,rangbuConj,shunchengConj,tiaojianConj,yinguoConj,zhuanzheConj,ConjNeg,ConjPosi,ConjAll'''.replace(
        '\n', '').replace(' ', '').split(',')

    data[Getname(doc_path)] = RWordPos(doc_path)
    pos_df = pd.DataFrame(data)
    return pos_df


def RPhrase(doc_path):
    '''
    外部调用
    '''
    trees = us.LoadToken(doc_path, 'tree')
    phrase_info = [us.TreNumLen(trees, pos_phrase) for pos_phrase in
                   ['VP', 'NP', 'ADVP', 'PP', 'CP', 'ADJP', 'ADVP', 'CLP', 'DNP', 'DVP', 'LCP']]
    phrase_info = [j for i in phrase_info for j in i]
    return phrase_info


def Phrase(doc_path):
    data = {}
    data['name'] = """VP_num,VPargv_phht,VPmax_phht,VPargv_phlen,VPargv_phlen,VPmax_phlen,
    NP_num,NPargv_phht,NPmax_phht,NPargv_phlen,NPargv_phlen,NPmax_phlen,
    ADVP_num,ADVPargv_phht,ADVPmax_phht,ADVPargv_phlen,ADVPargv_phlen,ADVPmax_phlen,
    PP_num,PPargv_phht,PPmax_phht,PPargv_phlen,PPargv_phlen,PPmax_phlen,
    CP_num,CPargv_phht,CPmax_phht,CPargv_phlen,CPargv_phlen,CPmax_phlen,
    ADJP_num,ADJPargv_phht,ADJPmax_phht,ADJPargv_phlen,ADJPargv_phlen,ADJPmax_phlen,
    ADV_num,ADVPargv_phht,ADVPmax_phht,ADVPargv_phlen,ADVPargv_phlen,AVPmax_phlen,
    CLP_num,CLPargv_phht,CLPmax_phht,CLPargv_phlen,CLPargv_phlen,CLPmax_phlen,
    DNP_num,DNPargv_phht,DNPmax_phht,DNPargv_phlen,DNPargv_phlen,DNPmax_phlen,
    DVP_num,DVPargv_phht,DVPmax_phht,DVPargv_phlen,DVPargv_phlen,DVPmax_phlen,
    LCP_num,LCPargv_phht,LCPmax_phht,LCPargv_phlen,LCPargv_phlen,LCPmax_phlen""".replace('\n', '').replace(' ',
                                                                                                           '').split(
        ',')
    data[Getname(doc_path)] = RPhrase(doc_path)
    syntree_df = pd.DataFrame(data)
    return syntree_df


def RSynTree(doc_path):
    '''
    外部调用
    '''
    trees = us.LoadToken(doc_path, 'tree')
    tree_height = [t.height() for t in trees]
    mean_th, max_th, min_th = us.StatInfo(tree_height)
    return mean_th, max_th, min_th


def SynTree(doc_path):
    data = {}
    data['name'] = 'mean_th,max_th,min_th'.split(',')

    data[Getname(doc_path)] = RSynTree(doc_path)
    syntree_df = pd.DataFrame(data)
    return syntree_df


def RDependency(doc_path):
    '''
    外部调用
    '''
    words = us.LoadToken(doc_path, 'word')
    sents = us.LoadToken(doc_path, 'sent')

    sents_relations = us.LoadDepen(doc_path)
    mainvs = [us.MainV(s) for s in sents_relations]
    mean_mainv, max_mainv, min_mainv = us.StatInfo(mainvs)
    modi_num, max_modilen, argv_modilen = us.Modi(sents_relations)

    mdds_sent = [us.MDD(s) for s in sents_relations]
    mdd_text = sum(mdds_sent) / (len(words) - len(sents)) if (len(words) - len(sents)) != 0 else 0

    mean_mdd, max_mdd, min_mdd = us.StatInfo(mdds_sent)

    return mean_mainv, max_mainv, min_mainv, modi_num, max_modilen, argv_modilen, mean_mdd, max_mdd, min_mdd, mdd_text


def Dependency(doc_path):
    data = {}
    data[
        'name'] = 'mean_mainv,max_mainv,min_mainv,modi_num,max_modilen,argv_modilen,mean_mdd,max_mdd,min_mdd,mdd_text'.split(
        ',')

    data[Getname(doc_path)] = RDependency(doc_path)
    depend_df = pd.DataFrame(data)
    return depend_df


def RPgComplex(doc_path):
    '''
    外部调用
    '''
    pgs = us.LoadToken(doc_path, 'pg')
    pg = len(pgs)
    mean_lenz_pg, max_lenz_pg, min_lenz_pg, mean_lenw_pg, max_lenw_pg, min_lenw_pg = us.CountLen(pgs)
    return pg, mean_lenz_pg, max_lenz_pg, min_lenz_pg, mean_lenw_pg, max_lenw_pg, min_lenw_pg


def PgComplex(doc_path):
    data = {}
    data['name'] = 'pg_num,mean_lenz_pg,max_lenz_pg,min_lenz_pg,mean_lenw_pg,max_lenw_pg,min_lenw_pg'.split(',')

    data[Getname(doc_path)] = RPgComplex(doc_path)
    pgcomplex_df = pd.DataFrame(data)
    return pgcomplex_df


def Pair(lis):
    '''
    内部
    '''
    prired_lis = ''.split(',')
    for i in range(len(lis) - 1):
        prired_lis.append((lis[i], lis[i + 1]))
    return prired_lis


def ReadPgPos(doc_path):
    '''
    内部
    '''
    # print(doc_path)
    root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    pos_path = os.path.join(root, 'corpus/pos')
    doc_name = doc_path.split('/')[-1]
    doc_path = os.path.join(pos_path, doc_name)

    doc_name = os.path.split(doc_path)[-1]
    corpus_name = os.path.split(os.path.split(doc_path)[0])[-1]
    corpus_path = os.path.join(os.getcwd(), 'corpus')
    pos_path = os.path.join(corpus_path, '{}-pos'.format(corpus_name))
    pos_path = os.path.join(pos_path, doc_name)

    name, word_pos_content = us.ReadDoc(pos_path)
    pg_pos = word_pos_content.split('\n\n')
    # pg_pos = [pg.split(' ') for pg in pg_pos]
    return pg_pos


def CountTokenOverlap(possent_1, possent_2):
    '''
    内部
    '''
    possent_1 = possent_1.split(' ')
    possent_2 = possent_2.split(' ')
    tokensent_1 = [i.split('_')[0] for i in possent_1]
    tokensent_2 = [i.split('_')[0] for i in possent_2]
    return len([i for i in tokensent_1 if i in tokensent_2])


def CountPosOverlap(possent_1, possent_2, poslis):
    '''
    内部
    '''
    possent_1 = possent_1.split(' ')
    possent_2 = possent_2.split(' ')
    contsent_1 = ''.split(',')
    contsent_2 = ''.split(',')
    for i in possent_1:
        if len(i.split('_')) > 1:
            if i.split('_')[1] in poslis:
                contsent_1.append(i.split('_')[0])
            else:
                pass
        else:
            pass
    for i in possent_2:
        if len(i.split('_')) > 1:
            if i.split('_')[1] in poslis:
                contsent_2.append(i.split('_')[0])
            else:
                pass
        else:
            pass

    contsent_1 = [i for i in contsent_1 if len(i) > 0]
    contsent_2 = [i for i in contsent_2 if len(i) > 0]
    overlap_num = len([i for i in contsent_1 if i in contsent_2])
    return overlap_num


def ROverlap(doc_path):
    '''
    外部调用
    '''
    pgs_pos = ReadPgPos(doc_path)
    pgs_pos = [s for s in pgs_pos if len(s) != 0]
    sents_pos = []
    # print(pgs_pos)
    ContentPos = ['a', 'b', 'i', 'j', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'q', 'r', 'v']
    NounPos = ['n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz']
    VPos = ['v']
    for pg in pgs_pos:
        sents = pg.split('\n')
        sents = [s for s in sents if len(s) != 0]
        sents_pos.extend(sents)

    if len(sents_pos) > 1:
        sent_pairs = Pair(sents_pos)
        sent_pairs = [i for i in sent_pairs if len(i) > 0]
        op_token_s = [CountTokenOverlap(possent_1, possent_2) for (possent_1, possent_2) in sent_pairs]
        op_content_s = [CountPosOverlap(possent_1, possent_2, ContentPos) for (possent_1, possent_2) in sent_pairs]
        op_n_s = [CountPosOverlap(possent_1, possent_2, NounPos) for (possent_1, possent_2) in sent_pairs]
        op_v_s = [CountPosOverlap(possent_1, possent_2, VPos) for (possent_1, possent_2) in sent_pairs]

        mean_op_token_s, max_op_token_s, min_op_token_s = us.StatInfo(op_token_s)
        mean_op_content_s, max_op_content_s, min_op_content_s = us.StatInfo(op_content_s)
        mean_op_n_s, max_op_n_s, min_op_n_s = us.StatInfo(op_n_s)
        mean_op_v_s, max_op_v_s, min_op_v_s = us.StatInfo(op_v_s)

    else:
        mean_op_token_s = max_op_token_s = min_op_token_s = mean_op_content_s = max_op_content_s = min_op_content_s = mean_op_n_s = max_op_n_s = min_op_n_s = mean_op_v_s = max_op_v_s = min_op_v_s = 0

    if len(pgs_pos) > 1:
        pg_pairs = Pair(pgs_pos)
        pg_pairs = [i for i in pg_pairs if len(i) > 0]

        op_token_pg = [CountTokenOverlap(possent_1, possent_2) for (possent_1, possent_2) in pg_pairs]
        op_content_pg = [CountPosOverlap(possent_1, possent_2, ContentPos) for (possent_1, possent_2) in pg_pairs]
        op_n_pg = [CountPosOverlap(possent_1, possent_2, NounPos) for (possent_1, possent_2) in pg_pairs]
        op_v_pg = [CountPosOverlap(possent_1, possent_2, VPos) for (possent_1, possent_2) in pg_pairs]

        mean_op_token_pg, max_op_token_pg, min_op_token_pg = us.StatInfo(op_token_pg)
        mean_op_content_pg, max_op_content_pg, min_op_content_pg = us.StatInfo(op_content_pg)
        mean_op_n_pg, max_op_n_pg, min_op_n_pg = us.StatInfo(op_n_pg)
        mean_op_v_pg, max_op_v_pg, min_op_v_pg = us.StatInfo(op_v_pg)
    else:
        mean_op_token_pg = max_op_token_pg = min_op_token_pg = mean_op_content_pg = max_op_content_pg = min_op_content_pg = mean_op_n_pg = max_op_n_pg = min_op_n_pg = mean_op_v_pg = max_op_v_pg = min_op_v_pg = 0

    return (mean_op_token_s, max_op_token_s, min_op_token_s,
            mean_op_content_s, max_op_content_s, min_op_content_s,
            mean_op_n_s, max_op_n_s, min_op_n_s,
            mean_op_v_s, max_op_v_s, min_op_v_s,
            mean_op_token_pg, max_op_token_pg, min_op_token_pg,
            mean_op_content_pg, max_op_content_pg, min_op_content_pg,
            mean_op_n_pg, max_op_n_pg, min_op_n_pg,
            mean_op_v_pg, max_op_v_pg, min_op_v_pg)


def Overlap(doc_path):
    data = {}
    data['name'] = """
    mean_op_token_s,max_op_token_s,min_op_token_s,
    mean_op_content_s,max_op_content_s,min_op_content_s,
    mean_op_n_s,max_op_n_s,min_op_n_s,
    mean_op_v_s,max_op_v_s,min_op_v_s,
    mean_op_token_pg,max_op_token_pg,min_op_token_pg,
    mean_op_content_pg,max_op_content_pg,min_op_content_pg,
    mean_op_n_pg,max_op_n_pg,min_op_n_pg,
    mean_op_v_pg,max_op_v_pg,min_op_v_pg""".replace('\n', '').replace(' ', '').split(',')
    data[Getname(doc_path)] = ROverlap(doc_path)
    overlap_df = pd.DataFrame(data)
    return overlap_df