import os,re
import pandas as pd
import numpy as np
from zhon import hanzi
from nltk.tree import Tree
#from lexicalrichness import LexicalRichness
from lexical_diversity import lex_div as ld
from itertools import groupby
from collections import Counter
import utils as us
from pandas.core.frame import DataFrame
def cut(data):
    phs = data
    sents = []
    punt_list = '!?。！？~'
    total_punc_list = '"[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"”'
    phs = phs.split()
    for ph in phs:
        length = len(ph)
        if length < 3:
            continue
        flag = 0
        start = 0
        i = 0
        j = 1
        while(j <= length - 1):
            word = ph[i]
            token = ph[j]
            if flag == 0:
                if  word in punt_list :
                    if token not in total_punc_list:# 检查标点符号下一个字符是否还是标点
                        sents.append(ph[start:j].strip())
                        start = j
                        i += 1
                        j += 1
                    else:
                        i += 1
                        j += 1
                        flag = 1
                else:
                    i += 1
                    j += 1
            else:
                if token not in total_punc_list:
                    sents.append(ph[start:j].strip())
                    start = j
                    j += 1
                    i += 1
                    flag = 0
                else:
                    i += 1
                    j += 1

        if start < len(ph) - 1:
            sents.append(ph[start:])
    return sents

def GetResourceRoot():
    return os.path.join(os.getcwd(),'resource')

def Writefile(path,content):
    fw = open(path,'w',encoding = 'utf-8')
    fw.write(content)
    fw.write('\n')
    fw.close()
    #print('{} ok'.format(path))
    return

def GetFile(dir_path):
    file_lis = []
    print(dir_path)
    list = os.listdir(dir_path) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):#因为文件夹里有一个莫名奇妙的desktop.ini，所以-1，正常的没有“-1”#-1
           path = os.path.join(dir_path,list[i])
           if os.path.isdir(path):
              file_lis.extend(GetFile(path))
           if os.path.isfile(path):
              file_lis.append(path)
    # print(file_lis)
    return file_lis

def ReadDoc(doc_path):
    '''
	func: 读取文本
	input: 待分析文本的文件名
	output: 课文标题（str），课文内容（str）'''
    # print('!!查验点',doc_path)
    with open(doc_path,'r',encoding='utf-8') as f:
        content = f.read().split('\n\n')
        doc_name = content[0].replace('#','').replace(' ','').replace(',','，')
        doc_content = '\n\n'.join(content[1:])
    return doc_name,doc_content

def mkdir(path):
    '''
    创建路径
    '''
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False

def LoadResource_Dic(ResPath,depent_facotr,indepent_facotr):
    '''
	func: 读取资源文件（如字频、笔画数）'''
    ResName_dic = {}
    data = pd.read_csv(ResPath,sep='\t',encoding='utf-8')
    # print(data)
    # print((data[depent_facotr]).values.tolist())
    # print((data[indepent_facotr]).values.tolist())
    ResName_dic = dict(zip((data[depent_facotr]).values.tolist(),(data[indepent_facotr]).values.tolist()))#修改，加了一个valuetolist
    # print(ResName_dic)
    return ResName_dic

def CountRes_Dic(analyed_lis,resource_path,depent_facotr,indepent_facotr):
    '''
    func: 把文本转化为具体的值，如笔画数，或字频
    input: ['你','好','！']
    output：[7,5,0]

    depent_facotr:汉字/词汇
    indepent_facotr:资源（笔画数/字频/词频）
    '''
    Resource_Dic = LoadResource_Dic(resource_path,depent_facotr,indepent_facotr)
    target_lis = []
    for item in analyed_lis:
        if item in Resource_Dic.keys():
            target_lis.append(Resource_Dic[item])
        else:
            target_lis.append(0)

    target_lis = [int(t) for t in target_lis if t==t]
    return target_lis

def LoadResource_Lis(ResPath):
    '''
	func: 读取词表资源文件（对称字表，成语词表...）'''
    with open(ResPath,encoding='utf-8') as f:
        txt = f.read()
        Resource_Lis = txt.split(",")
    return Resource_Lis

def CountRes_Lis(From_Lis,ResPath):
    '''
    用计算对称字或成语数之类的 list形式的资源
    input:['一','二','哈']，资源路径
    output：['一','二'] 
    '''
    Resource_Lis = LoadResource_Lis(ResPath)
    return [i for i in From_Lis if i in Resource_Lis]

def FilterHZ(token_lis):
    '''
    func：去除分好词的词表中的标点。
    input ['你好'，'！']
    return ['你好']
    '''
    lis = [''.join(re.findall('[%s]' % hanzi.characters,t)) for t in token_lis]
    return [i for i in lis if len(i)>0]

def LoadToken(doc_path,formate):
    '''
    func：输入文档路径，输出分字列表、分词列表、分句列表、分段落列表
    formate: char,token,sent,pg
    '''
    # print(doc_path)
    doc_name = os.path.split(doc_path)[-1]
    corpus_name = os.path.split(os.path.split(doc_path)[0])[-1]
    corpus_path = os.path.join(os.getcwd(),'corpus')
    # print(doc_name,corpus_path,corpus_name)
    if formate == 'char':
        # print('aaaa', doc_path)
        txt_name,txt_content = ReadDoc(doc_path)
        regex=re.compile(".")
        zi_lis = regex.findall(txt_content)
        tokens = FilterHZ(zi_lis)

    if formate == 'word':
        token_path = os.path.join(corpus_path,'{}-token'.format(corpus_name))
        path = os.path.join(token_path,doc_name)
        txt_name,txt_content = ReadDoc(path)
        tokens = txt_content.split(' ')
        tokens = FilterHZ(tokens)

    if formate == 'sent':
        token_path = os.path.join(corpus_path,'{}-token'.format(corpus_name))
        path = os.path.join(token_path,doc_name)
        txt_name,txt_content = ReadDoc(path)
        tokens = txt_content.split('\n')

    if formate == 'pg':
        token_path = os.path.join(corpus_path,'{}-token'.format(corpus_name))
        path = os.path.join(token_path,doc_name)
        txt_name,txt_content = ReadDoc(path)
        tokens = txt_content.split('\n\n')

    if formate == 'pos':
        pos_path = os.path.join(corpus_path,'{}-pos'.format(corpus_name))
        path = os.path.join(pos_path,doc_name)
        with open(path,'r',encoding='utf8') as f:
            content = f.read().split('\n\n')
            content = [i.replace('\n',' ').split(' ') for i in content[1:]]
        tokens = [j for i in content for j in i] 

    if formate == 'ner':
        pos_path = os.path.join(corpus_path,'{}-ner'.format(corpus_name))
        path = os.path.join(pos_path,doc_name)
        with open(path,'r',encoding='utf8') as f:
            content = f.read().split('\n\n')
            #name = content[0]
            content = [i.split(' ') for i in content[1:]]
        #name,word_ner_content = ReadDoc(doc_path)
        tokens = [j for i in content for j in i] 
        
    if formate == 'tree':
        tree_path = os.path.join(corpus_path,'{}-tree'.format(corpus_name))
        path = os.path.join(tree_path,doc_name)
        tokens = []
        with open(path,'r',encoding='utf-8') as f:
            content = f.read().split('\n\n')
            pgs = content[1:]
            for pg in pgs:
                pg = pg.split('\n\n')
                pg = [i for i in pg if len(i)!=0]
                for sent in pg:
                    #print(sent)
                    sent = sent.replace('\n','')
                    if sent.find('(ROOT') != -1:
                        try:
                            Tre = Tree.fromstring(sent)
                            tokens.append(Tre)
                        except ValueError as err:
                            print('exception',err)
    return [i for i in tokens if len(i)>0]


def CountDoc(word_tokens,word_type):
    typo_docfreq = []
    for typo in word_type:
        doc_freq = 0
        for tokens in word_tokens:
            if typo in tokens:
                doc_freq+=1
        typo_docfreq.append(doc_freq)
    return typo_docfreq


def DocParser(corpus_path):
    corpus_name = os.path.split(corpus_path)[-1]

    files_lis = us.GetFile(corpus_path)
    # print('查验点!',files_lis)
    # print(' '.join((os.path.split(file)[-1]) for file in files_lis))
    word_tokens = [list(set(us.LoadToken(file,'word'))) for file in files_lis]#修改，原为word_tokens = [list(set(us.LoadToken(os.path.split(file)[-1],'word'))) for file in files_lis]，结果加载出来的并不是路径，而只是文件名
    word_type = list(set([j for i in word_tokens for j in i]))
    wordtypo_docfreq = CountDoc(word_tokens,word_type)

    wf_df = pd.DataFrame()
    wf_df['Word'] = word_type
    wf_df['Docfreq'] = wordtypo_docfreq
    word_topath = os.path.join(GetResourceRoot(),'{}-worddocfreq.txt'.format(corpus_name))
    wf_df.to_csv(word_topath,sep='\t',encoding='utf8',index=False)
    print('Ready: words doc freq ')

    char_tokens = [list(set(us.LoadToken(file,'char'))) for file in files_lis]#修改
    char_type = list(set([j for i in char_tokens for j in i]))
    chartypo_docfreq  = CountDoc(char_tokens,char_type)
    cf_df = pd.DataFrame()
    cf_df['Char'] = char_type
    cf_df['Docfreq'] = chartypo_docfreq
    char_topath = os.path.join(GetResourceRoot(),'{}-chardocfreq.txt'.format(corpus_name))
    cf_df.to_csv(char_topath,sep='\t',encoding='utf8',index=False)
    print('Ready: chars doc freq ')

    return

def StatInfo(stat_lis):
    '''
    func：得到数字列表的均值和最大值
    input [2,2,4,2]
    return [2.5,4]
    '''
    if len(stat_lis) > 0:
        argv = np.mean(stat_lis)
        maxi = max(stat_lis)
        mini = min(stat_lis)
    else:
        argv = maxi = mini = 0
    return [argv,maxi,mini]

def ReturnPY(analyed_lis,resource_path,depent_facotr,indepent_facotr):
    Resource_Dic = LoadResource_Dic(resource_path,depent_facotr,indepent_facotr)
    target_lis = []
    for item in analyed_lis:
        if item in Resource_Dic.keys():
            target_lis.append(Resource_Dic[item])
        else:
            target_lis.append('a')

    target_lis = [t for t in target_lis if t==t]

    return target_lis

def CountWordPY(word):
    hz_info_path = os.path.join(GetResourceRoot(),'Hanzinfo.txt')
    word_lens_py = len(list(''.join(ReturnPY(list(word),hz_info_path,'Char','Pinyin'))))
    return word_lens_py

def CountWordStroke(word):
    bihua_path = os.path.join(GetResourceRoot(),'HZ2Bihua.txt')
    zi_bihua = CountRes_Dic(list(word),bihua_path,'Char','Stroke')
    wordlen_stroke = sum(zi_bihua)
    return wordlen_stroke

def CountLevel(doc_path,mode,corpus_path,depent_facotr,indepent_facotr):

    tokens = LoadToken(doc_path,mode)
    dl_token = CountRes_Dic(tokens,corpus_path,depent_facotr,indepent_facotr)
    types = list(set(tokens))
    dl_type = CountRes_Dic(types,corpus_path,depent_facotr,indepent_facotr)
    argv_dl_token = np.mean(dl_token)
    var_dl_token = np.var(dl_token)

    levels = sorted(set(dl_token))
    if "HGJChar1" in corpus_path or "OldHSKChar1" in corpus_path or "OldHSKWord1" in corpus_path:
        origin_levels = [i for i in range(5)]
        counter_token = dict(Counter(dl_token))
        counter_type = dict(Counter(dl_type))
        for origin_level in origin_levels:
            if origin_level not in counter_token.keys():
                counter_token[origin_level]=0
                counter_type[origin_level]=0
        eachlevel_tokennum = counter_token.values()
        eachlevel_typennum = counter_type.values()
    if "NewHSKChar1" in corpus_path or "HGJWord1" in corpus_path or "NewHSKWord1" in corpus_path:
        origin_levels = [i for i in range(7)]
        counter_token = dict(Counter(dl_token))
        counter_type = dict(Counter(dl_type))
        for origin_level in origin_levels:
            if origin_level not in counter_token.keys():
                counter_token[origin_level]=0
                counter_type[origin_level]=0
        eachlevel_tokennum = counter_token.values()
        eachlevel_typennum = counter_type.values()
    if "HuayuWordr1" in corpus_path:
        origin_levels = [i for i in range(8)]
        counter_token = dict(Counter(dl_token))
        counter_type = dict(Counter(dl_type))
        for origin_level in origin_levels:
            if origin_level not in counter_token.keys():
                counter_token[origin_level]=0
                counter_type[origin_level]=0
        eachlevel_tokennum = counter_token.values()
        eachlevel_typennum = counter_type.values()
    counter_token = [(k,counter_token[k]) for k in sorted(counter_token.keys())] 
    counter_type = [(k,counter_type[k]) for k in sorted(counter_type.keys())] 
    eachlevel_tokennum = [i[1] for i in counter_token]
    eachlevel_typennum = [i[1] for i in counter_type]
    data = [argv_dl_token,var_dl_token]+eachlevel_tokennum+eachlevel_typennum
    return data

def CountDiff(tokens,resource_path,depent_facotr,numbers):
    diff_df = pd.read_csv(resource_path,sep='\t',encoding="utf8")
    diff_tokens = diff_df[depent_facotr].tolist()[:numbers]

    diff_token_num = len([i for i in tokens if i in diff_tokens])
    diff_type_num = len([i for i in list(set(tokens)) if i in diff_tokens])

    return diff_token_num,diff_type_num

def TTR(tokens):
    #print('flt:{}'.format(flt))
    if len(tokens)>0:
        typo = len(set(tokens))
        ttr = ld.ttr(tokens)
        root = ld.root_ttr(tokens)
        log = ld.log_ttr(tokens)
        mass = ld.maas_ttr(tokens)
        msttr = ld.msttr(tokens)
    else:
        typo=ttr=root=log=mass=msttr=0
    return typo,ttr,root,log,mass,msttr

def FreqWeight(int_lis,tokens,freq_resource_path,depent_facotr,indepent_facotr):
    freuqency = CountRes_Dic(tokens,freq_resource_path,depent_facotr,indepent_facotr)
    freq_weighted_bihua = sum([i*j for i in int_lis for j in freuqency ])/sum(freuqency) if sum(freuqency) != 0 else 0
    return freq_weighted_bihua


def Freqencey(doc_path,mode,corpus_path,depent_facotr,indepent_facotr):
    tokens = LoadToken(doc_path,mode)
    types = list(set(tokens))
    freq_tokens = CountRes_Dic(tokens,corpus_path,depent_facotr,indepent_facotr)
    freq_types= CountRes_Dic(types,corpus_path,depent_facotr,indepent_facotr)
    # print(freq_types,'\n',freq_tokens)
    argv_corf_token,max_corf_token,min_corf_token = StatInfo(freq_tokens)
    argv_corf_type,max_corf_type,min_corf_type = StatInfo(freq_types)
    argv_logf_token,max_logf_token,min_logf_token = StatInfo([np.log(f) for f in freq_tokens if np.log(f) != float("-inf")])
    argv_logf_type,max_logf_type,min_logf_type = StatInfo([np.log(f) for f in freq_types if np.log(f) != float("-inf")])
    oov_corf = freq_tokens.count(0)
    freqs = [oov_corf,
        argv_corf_token,max_corf_token,
        argv_corf_type,max_corf_type,
        argv_logf_token,max_logf_token,
        argv_logf_type,max_logf_type,]
    return freqs

def ContentFreq(doc_path,mode,corpus_path,depent_facotr,indepent_facotr):
    word_pos = LoadToken(doc_path,mode,)
    content_tokens = ReturnContent(word_pos)

    freq_content_tokens = CountRes_Dic(content_tokens,corpus_path,depent_facotr,indepent_facotr)
    freq_content_types= CountRes_Dic(list(set(content_tokens)),corpus_path,depent_facotr,indepent_facotr)
    argvfreq_content_tokens = np.mean(freq_content_tokens)
    argvfreq_content_types = np.mean(freq_content_types)
    return [argvfreq_content_tokens,argvfreq_content_types]

def ReturnContent(word_pos):
    ContentPos = ['a','b','i','j','m','n','nd','nh','ni','nl','ns','nt','nz','q','r','v']
    content_word = []
    for i in word_pos:
        if len(i.split('_'))==2:
            if i.split('_')[1] in ContentPos:
                content_word.append(i.split('_')[0])
        else:
            print(i)
    #content_word = [i.split('_')[0] for i in word_pos if i.split('_')[1] in ContentPos]
    return content_word

def CountSpecical(doc_path,mode,resource_path,colname):
    tokens = LoadToken(doc_path,mode)
    specials_df = pd.read_csv(resource_path)
    # print(specials_df)
    specials = specials_df[colname].tolist()
    special_len = len([i for i in tokens if i in specials])
    # print(special_len)
    return special_len

def CountSpecical_(doc_path,mode,resource_path,colname):
    tokens = LoadToken(doc_path,mode)
    specials_df = pd.read_excel(resource_path,engine='openpyxl')

    c = {colname: tokens}
    df1 = DataFrame(c)
    df2 = pd.merge(df1, specials_df, on= colname, how='inner')
    array_data = np.array(df2)  # df数据转为np.ndarray()
    list_data = array_data.tolist()  # 将np.ndarray()转为列表
    special_len=len(list_data)
    # print(special_len)
    return special_len

def CountSpecical__(doc_path,mode,resource_path,depent_facotr, indepent_facotr):
    tokens = LoadToken(doc_path, mode)
    Resource_Dic = LoadResource_Dic(resource_path, depent_facotr, indepent_facotr)
    target_lis = []
    for item in tokens:
        if item in Resource_Dic.keys():
            target_lis.append(Resource_Dic[item])
    # target_lis = [int(t) for t in target_lis if t == t]
    argv, maxi, mini = StatInfo(target_lis)
    vari = np.var(target_lis)
    sumresult = sum(target_lis)
    return argv, maxi, mini, vari,sumresult

def SpecicalWord_dic(doc_path,mode,resource_path,depent_facotr,indepent_facotr):
    tokens = LoadToken(doc_path,mode)
    Word_Meaning = CountRes_Dic(tokens,resource_path,depent_facotr,indepent_facotr)
    # print(Word_Meaning)
    word_meaning = [i for i in Word_Meaning if i != 0]
    if len(word_meaning) > 0:
        argv,maxi,mini = StatInfo(word_meaning)
        vari = np.var(word_meaning)
    else:
        argv=maxi=mini=vari=0
    return argv,maxi,mini,vari

def CountPos(word_pos,specific_pos):
    if isinstance(specific_pos,list):
        pos = [i.split('_')[0] for i in word_pos if i.split('_')[1] in specific_pos]
    elif isinstance(specific_pos,str):
        pos = [i.split('_')[0] for i in word_pos if i.split('_')[1]==specific_pos]
    return len(pos)


def CountUnion(lis01,lis02):
	return [i for i in lis01 if i in lis02]

def CountLen(lis):
    len_z = [len(re.findall('[%s]' % hanzi.characters,l)) for l in lis] 
    len_w = [len(l.split(' ')) for l in lis]
    mean_lenz,max_lenz,min_lenz = StatInfo(len_z)
    mean_lenw,max_lenw,min_lenw = StatInfo(len_w)
    return mean_lenz,max_lenz,min_lenz,mean_lenw,max_lenw,min_lenw

def EasyComplex(sents):
    '''
    内部
    '''
    easy_sent = 0
    complex_sent = 0
    for s in sents:
        if s.find('，') != -1 or s.find('；') != -1 or s.find('：') != -1:
            complex_sent+=1
        else:
            easy_sent +=1
    return easy_sent,complex_sent

def CountSentPos(doc_path):
    doc_name = os.path.split(doc_path)[-1]
    corpus_path = os.path.join(os.getcwd(),'corpus')
    corpus_name = os.path.split(os.path.split(doc_path)[0])[-1]
    pos_path = os.path.join(corpus_path,'{}-pos'.format(corpus_name))
    path = os.path.join(pos_path,doc_name)

    txt_name,txt_content = ReadDoc(path)
    sents = [i for i in txt_content.split('\n') if len(i)>0]
    sent_posnum = []
    for s in sents:
        poses = s.split(' ')
        pos_num = len(set([p.split('_')[1] for p in poses]))
        sent_posnum.append(pos_num)
    mean,maxi,mini = StatInfo(sent_posnum)
    return mean,maxi,mini

def ReadParser(tree_path):
    '''
    内部
    '''
    with open(tree_path,'r',encoding='utf-8') as f:
        content = f.read().split('\n\n')
        doc_name = content[0]
        doc_content = '\n\n\n'.join(content[1:])
    return doc_name,doc_content

def PhraseTree(tree,phrase_name):
    cadidate = []
    cadidate_tree = []
    phrases = tree.subtrees(lambda t: t.label() == phrase_name)

    for phrase in phrases:
        leaves = ' '.join(phrase.leaves())
        #print(leaves)
        candi_str = ' '.join(cadidate)
        if leaves not in candi_str:
            cadidate.append(leaves)
            cadidate_tree.append(phrase)
        else:
            continue
    return cadidate_tree

def TreNumLen(trees,phrase_name):
    phrase_trees = []
    for tree in trees:
        t = PhraseTree(tree,phrase_name)
        if len(t)>0:
            phrase_trees.append(t)
    
    if len(phrase_trees)>0:
        phrase_trees = [j for i in phrase_trees for j in i]
        phrase_heightlis = [i.height() for i in phrase_trees]
        phrase_lenlis = [len(i) for i in phrase_trees]
        
        phrase_num = len(phrase_trees)
        argv_phht = np.mean(phrase_heightlis)
        max_phht = max(phrase_heightlis)
        argv_phlen = np.mean(phrase_lenlis)
        max_phlen = max(phrase_lenlis)
    else:
        phrase_num=argv_phht=max_phht=argv_phlen=argv_phlen=max_phlen=0
    
    return phrase_num,argv_phht,max_phht,argv_phlen,argv_phlen,max_phlen

def LoadDepen(doc_path):
    '''
    内部
    '''
    doc_name = os.path.split(doc_path)[-1]
    corpus_path = os.path.join(os.getcwd(),'corpus')
    corpus_name = os.path.split(os.path.split(doc_path)[0])[-1]
    parser_path = os.path.join(corpus_path,'{}-parser'.format(corpus_name))
    parser_path = os.path.join(parser_path,doc_name)
    
    doc_name,doc_content = ReadParser(parser_path)
    pgs = doc_content.split('\n\n\n')  
    sents_relations = ''.split(',')
    for pg in pgs:
        sent_relations = pg.split('\n\n')
        sents_relations.extend(sent_relations)
    return sents_relations

def MainV(sent_relations):
    '''
    内部
    '''
    relations = sent_relations.split('\n')
    relations = [r for r in relations if len(r)>0]
    mainvs = [relations.index(r) for r in relations if r.find('HED') != -1]
    if len(mainvs) >0:
        mv = mainvs[0]
    else:
        mv = 0
    return mv

def MDD(sent_relations):
    '''
    内部
    '''
    relations = sent_relations.split('\n')
    dd = 0
    for r in relations:
        rs = r.split(' ')
        if len(rs) == 3:
            if rs[2] != 'HED' and rs[2] != 'WP':
                word_1 = rs[0].split('_')
                word_1_loc = word_1[1]
                word_2 = rs[1].split('_')
                word_2_loc = word_2[1]
                dd += abs(int(word_2_loc) - int(word_1_loc))
    mdd = dd/(len(relations)-2) if (len(relations)-2) >0 else 0
    return mdd

def Modi(sents_relations):
    sents_relations = [i for i in sents_relations if len(i)>0 and i != "\n"]
    
    all_relation = []
    continue_modis = []
    for sent_relations in sents_relations:
        #print(sent_relations)
        relation = []
        relations = sent_relations.split('\n')
        for r in relations:
            rs = r.split(' ')
            #print(rs)
            if len(rs) == 3:
                all_relation.append(rs[2])
                relation.append(rs[2])
        #print(relation)
        modi_lens = [len(list(g)) for k, g in groupby(relation) if k=='ATT' or k=='ADV']
        if len(modi_lens)>0:
            continue_modi = max(modi_lens)
            continue_modis.append(continue_modi)
    att_num = all_relation.count('ATT')
    adv_num = all_relation.count('ADV')
    if len(continue_modis)>0:
        max_modi = max(continue_modis)
        argv_modi = np.mean(continue_modis)
    else:
        max_modi=argv_modi=1
    return att_num+adv_num,max_modi,argv_modi

def LoadWordlis(wordlis_path):
    worddf = pd.read_csv(wordlis_path,sep='\t',encoding="utf-8")
    lis = []
    for col in list(worddf.columns):
        lis.append(worddf[col].tolist())
    return lis