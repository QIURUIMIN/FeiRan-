# -*- coding = utf-8 -*-
# @Time : 2021/7/1 12:12
# @Author : ly
# @File : featanalyzeall.py
# @Software: PyCharm
import os, sys, time
import pandas as pd

root_dir = os.getcwd()
sys.path.append(os.path.join(root_dir, 'code'))
import feature
import parser_ly

current_dir = os.getcwd()  # obtain work dir
sys.path.append(current_dir)  # add work dir to sys path
import utils as us

if sys.version_info[0] >= 3:
    xrange = range


def exit_with_help(argv):#修改
    print("""\
Usage: {0} functions corpuspath output

functions: the txtfile that restore multi functions
corpuspath: the dirpath of corpus
output:  the filename of the output data.""".format(argv[0]))
    exit(1)


def ReadFunc(functions_name):
    functions_path = os.path.join(os.getcwd(), functions_name)
    with open(functions_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        functions = content.split('\n')
    return functions


def process_options(argv):#修改
    argc = len(argv)
    if argc < 3:
        exit_with_help(argv)

    functions_name = argv[1]
    functions = ReadFunc(functions_name)

    corpus_path = argv[2]
    output_file = argv[3]
    '''
    if os.path.isdir(corpus_path) == False:
        print ("corpus must be a directory")
        exit_with_help(argv)

    if os.path.isdir(corpus_path):
        print ("output file must be a file, please enter a filename")
        exit_with_help(argv)
    '''

    return functions, corpus_path, output_file


def ReadTxt(txt_path):
    with open(txt_path, 'r', encoding='utf8') as f:
        content = f.read().strip()
        content_lis = content.split('\n\n')
        txt_name = content_lis[0].replace('#', '').replace(' ', '').replace(',', '，')
        txt_content = '\n\n'.join(content_lis[1:])
    return txt_name, txt_content


def Proces_df(df):
    df.columns = df.iloc[0]
    df = df.drop(['name'])
    # df = df.ix[:,~((df==1).all())]
    print()
    print(df.info())
    print(df)
    return df


def main(argv=sys.argv):
    # functions, corpus_path, output_file = process_options(argv)#修改
    functions = ['TXT','Strokes','Sym_once','CharFreq','DocFreq','OOV','DiffLevelChar','DiffLevelWord','DiffChar','DiffWord','CommonChar','TTR','WordLen','WordFreq','WordOther','synmeaning','wcmeaning','SentComplex','WordPos','Phrase','SynTree','Dependency','PgComplex','Overlap','wencaidic']
    corpus_path = 'add'
    output_file = 'test.csv'

    corpus_path = os.path.join(os.getcwd(), 'corpus\\{}'.format(corpus_path))
    if 'DocFreq' in functions:
        # for mode in ['char', 'word']:
        us.DocParser(corpus_path)#修改us.DocParser(corpus_path, mode)#这一步应该是分开算char、word，合并起来了。

    timing = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('dealing... {}'.format(timing))

    files = os.listdir(corpus_path)
    files = [os.path.join(corpus_path, i) for i in files if i.endswith('.txt')]

    featdata = pd.DataFrame()

    for i, file_path in enumerate(files):
        print('dealing:{}\tdone:{}\ttotal:{}\t{}'.format(os.path.split(file_path)[-1], i, len(files), timing))

        data = pd.DataFrame()
        for function in functions:
            func_data = getattr(feature, function)(file_path)
            data = data.append(func_data)

        featdata = featdata.append(data.T)

    featdata = Proces_df(featdata)

    timing = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    output_path = os.path.join(os.getcwd(), 'output\\{}'.format(output_file))
    print('writing... {}\t{}'.format(output_file, timing))
    # featdata.rename(columns={'name':'txtname'})
    featdata.to_csv(output_path, encoding="utf-8-sig", index=True)
    print('Successfully Analyze')

    return


if __name__ == '__main__':
    main(sys.argv)
