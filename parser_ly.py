# -*- coding = utf-8 -*-
# @Time : 2021/7/1 12:12
# @Author : ly
# @File : parser.py
# @Software: PyCharm
# !/usr/bin/env python3#在此创建了一个虚拟环境
'''
预处理的代码

分析的语料是文件夹内的多个文本
文本由标题和文本组成，格式为：
‘
##标题

正文
’

经过五个自然语言处理步骤：
ltp：分词（token）、词性标注（pos）、命名实体识别（ner）、依存句法分析（parser）
Stanford parser：短语句法分析（tree）
每个处理结果存到一个文件夹内
'''

import os, sys
import time
import numpy as np
current_dir = os.getcwd()  # obtain work dir
sys.path.append(os.path.join(current_dir, 'code'))  # add work dir to sys path
import utils as us
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('D:\\desktop\\stanford-corenlp-4.2.2\\stanford-corenlp-4.2.2',lang="zh")
#nlp = StanfordCoreNLP('C:\\Users\\Li Yi\\anaconda3\\stanford-corenlp-4.2.2\\stanford-corenlp-4.2.2', lang="zh")
from nltk.tree import Tree


# from stanfordcorenlp import StanfordCoreNLP

def exit_with_help(argv):
    print("""\
Usage: {0} function corpuspath output

corpuspath: the dirpath of corpus
output:  the filename of the output data.""".format(argv[0]))
    exit(1)


def process_options(argv):
    argc = len(argv)
    if argc < 1:
        exit_with_help(argv)

    corpus_name = argv[1]
    corpus_path = os.path.join(os.getcwd(), 'corpus')
    corpus_path = os.path.join(corpus_path, corpus_name)
    return corpus_path


def ReadDoc(doc_path):
    '''
    func: 读取文本
    input: 待分析文本的文件名
    output: 课文标题（str），课文内容（str）'''
    with open(doc_path, 'r', encoding='utf-8') as f:
        # content = f.readlines()
        content = f.read().replace(',', '，').split('\n\n')
        # print(content)
        doc_name = content[0].replace('#', '')
        doc_content = '\n\n'.join(content[1:])
        # print('name:',doc_name,'content:',doc_content)
    return doc_name,doc_content#此部分要更改


def LTP(ltp_dir, pg):
    '''
    输入一个段落
    输出LTP对这段话的分析结果，包括：分词、词性标注、命名实体识别、依存句法分析和语义角色标注
    段落里句与句之间 分行
    '''
    # ltp模型目录的路径

    from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser

    '''载入模型'''
    cws_model_path = os.path.join(ltp_dir, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    pos_model_path = os.path.join(ltp_dir, 'pos.model')
    ner_model_path = os.path.join(ltp_dir, 'ner.model')
    par_model_path = os.path.join(ltp_dir, 'parser.model')

    sents = us.cut(pg)
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    # segmentor = Segmentor(cws_model_path)#修改

    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    # postagger = Postagger(pos_model_path)

    recognizer = NamedEntityRecognizer() # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型
    # recognizer = NamedEntityRecognizer(ner_model_path)

    parser = Parser() # 初始化实例
    parser.load(par_model_path)  # 加载模型
    # parser = Parser(par_model_path)

    words_pg = []
    postags_pg = []
    netags_pg = []
    parsers_pg = []
    for s in sents:
        words = segmentor.segment(s)
        tokens = ' '.join(list(words))
        words_pg.append(tokens)  # 分词 list
        # print(words_pg)
        postags = postagger.postag(words)  # 词性标注 list
        poses = ' '.join(["{}_{}".format(i[0], i[1]) for i in list(zip(words, postags))])
        postags_pg.append(poses)
        # print(postags_pg)
        ners = recognizer.recognize(words, postags)
        ner = ' '.join(["{}_{}".format(i[0], i[1]) for i in list(zip(words, ners))])
        netags_pg.append(ner)  # 命名实体识别 list
        # print(netags_pg)
        arcs = parser.parse(words, postags)  # 句法分析
        dp_result = [(arc.head, arc.relation) for arc in arcs]
        dps = ['{}_{} {}_{} {}'.format(words[i], str(i), words[int(dp_result[i][0]) - 1], int(dp_result[i][0]) - 1,
                                       dp_result[i][1]) for i in range(len(words))]
        parsers_pg.append('\n'.join(dps))

    segmentor.release()
    postagger.release()
    recognizer.release()
    parser.release()

    token_word = '\n'.join(words_pg)
    pos_word = '\n'.join(postags_pg)
    ner_word = '\n'.join(netags_pg)
    parser_doc = '\n\n'.join(parsers_pg)

    return token_word, pos_word, ner_word, parser_doc


def Process(LTP_DATA_DIR, doc_content):
    praph = doc_content.split('\n\n')

    ltp_result = [LTP(LTP_DATA_DIR, p) for p in praph]

    doc_token = '\n\n'.join([r[0] for r in ltp_result])
    doc_pos = '\n\n'.join([r[1] for r in ltp_result])
    doc_ner = '\n\n'.join([r[2] for r in ltp_result])
    doc_pars = '\n\n\n'.join([r[3] for r in ltp_result])

    return [doc_token, doc_pos, doc_ner, doc_pars]


def LTP_parser(files_dir, doc_name, ltp_dirs):#修改
    # print(ltp_dirs)
    # print(doc_name)

    # LTP_DATA_DIR = r'D:\ltp\ltp_data_v3.4.0'
    LTP_DATA_DIR = r'D:\\desktop\\ltp\\ltp_data'
    name, content = ReadDoc(os.path.join(files_dir, doc_name))
    LTP_result = Process(LTP_DATA_DIR, content)

    all_path = [os.path.join(dir_path, doc_name) for dir_path in ltp_dirs]

    for i, j in enumerate(all_path):
        # print(j)
        us.Writefile(j, '##{}\n\n{}'.format(name, LTP_result[i]))

    return


def BuildTree(seg_sent):
    '''输入分好词的句子，输出句法树。
    标记说明：https://blog.csdn.net/lihaitao000/article/details/51556923
    '''
    try:
        tree = nlp.parse(seg_sent)
        tree = tree.replace('\n', '')
        return tree
    except:
        print(seg_sent)
        return ' '


def Stanford_parser(token_dir, doc_name, tree_dir):
    # print(token_dir,doc_name,tree_dir)
    name, content = ReadDoc(os.path.join(token_dir, doc_name))
    pgs = content.split('\n\n')
    trees_pgs = []
    for pg in pgs:
        # print(pg)
        sents = [s for s in pg.split('\n') if len(s) > 2]
        trees_pg = [BuildTree(sent) for sent in sents]
        trees_pg = '\n\n'.join(trees_pg)
        trees_pgs.append(trees_pg)

    TREE = '\n\n\n'.join(trees_pgs)
    to_path = os.path.join(tree_dir, doc_name)
    us.Writefile(to_path, '##{}\n\n\n{}'.format(name, TREE))

    return


def Parser(files_dir):
    dir_path = os.path.split(files_dir)[0]
    token_dir = os.path.join(dir_path, '{}-token'.format(os.path.split(files_dir)[1]))
    pos_dir = os.path.join(dir_path, '{}-pos'.format(os.path.split(files_dir)[1]))
    ner_dir = os.path.join(dir_path, '{}-ner'.format(os.path.split(files_dir)[1]))
    parser_dir = os.path.join(dir_path, '{}-parser'.format(os.path.split(files_dir)[1]))
    tree_dir = os.path.join(dir_path, '{}-tree'.format(os.path.split(files_dir)[1]))
    dirs = [token_dir, pos_dir, ner_dir, parser_dir, tree_dir]
    ltp_dirs = [token_dir, pos_dir, ner_dir, parser_dir]
    # print(ltp_dirs)
    for d in dirs:
        us.mkdir(d)

    all_files = us.GetFile(files_dir)
    # print(all_files[:10])
    all_files = [i for i in all_files if i.endswith('.txt')]
    # pt_files = sorted([i.split('/')[-1] for i in all_files if i.find('pt') != -1])
    print('共读入文本{}篇'.format(len(all_files)))

    n = 0
    for doc_name in all_files:
        doc_name = os.path.split(doc_name)[-1]
        n += 1
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('正在处理第{}篇：{}，还剩{}篇，{}'.format(n, doc_name, len(all_files) - n, now_time))
        LTP_parser(files_dir, doc_name, ltp_dirs)
        print('分词、词性标注、命名实体识别、依存句法分析√')
        # print(token_dir)
        Stanford_parser(token_dir, doc_name, tree_dir)
        print('短语句法分析√')
    return


def main(argv=sys.argv):
    # files_path = process_options(argv)
    files_path = r'corpus\\add'
    Parser(files_path)  # 绝对路径
    return


if __name__ == '__main__':
    main(sys.argv)
