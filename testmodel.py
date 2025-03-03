# -*- coding = utf-8 -*-
# @Time : 2021/10/29 13:07
# @Author : ly
# @File : testmodel.py
# @Software: PyCharm

import featanalyzeall as analyze
import ProcessFeat as pro
"""

# testtxt=input('文本：')
# docname=input('name:')
# print(testtxt)
# print(docname)
# #
# txtcontent=testtxt.split('\\n')#这里因为是input，所以用了转换符。正常应该不用
# print(txtcontent)
# doc_content = '\n\n'.join(s for s in txtcontent if len(s) != 0)
# newcontent='##'+docname+'\n\n'+doc_content
# print(newcontent)
# n = ('H:\\code\\corpus\\test\\' +docname+'.txt')
# file = open(n, 'w', encoding='utf-8',buffering=0)
# file.write(newcontent)

files_path = r'corpus\\add'

import parser_ly as parser
parser.Parser(files_path)
analyze.main()
featdata_name=r'model\\output\\test.csv'
pro.deal(featdata_name)

import pandas as pd

df1 = pd.read_excel('output\\liyi_3_1.xlsx', engine='openpyxl')
df2=pd.read_csv('output\\liyi.csv')
df3=pd.merge(df1,df2,on='txtname', how='outer')
# print(df3)
common_columns = list(set(list(df1.columns)) & set(list(df2.columns)))
common_columns.remove('txtname')
# common_columns
for col in common_columns:
    df3[col] = df3[col+'_x'].where(df3[col+'_x'].notnull(), df3[col+'_y'])
clash_names = [elt+suffix for elt in common_columns for suffix in ('_x','_y') ]
df3.drop(labels=clash_names, axis=1,inplace=True)
print(df3)
df3.to_csv('output\\test1.csv',index=False,sep=',')
"""
#读取模型
import pandas as pd
import joblib
dirs='wencai_model.m'
SVM=joblib.load(dirs)
df=pd.read_csv("output/test1.csv")
#isNA=df.isnull()
#print(df[isNA.any(axis=1)])



X=df[['dl02_tokenchar_hgj_rate','var_dl_tokenchar_oldhsk_rate','dl01_tokenchar_newhsk_rate','dl04_typeword_oldhsk','oov_tokenchar_hgj_rate','var_dl_tokenchar_newhsk_rate','oov_tokenword_oldhsk_rate','dl06_tokenchar_newhsk_rate','oov_tokenchar_oldhsk_rate','dl02_tokenchar_oldhsk_rate','var_dl_tokenword_hgj_rate','dl05_tokenword_hgj_rate','oov_tokenword_newhsk_rate','oov_tokenchar_newhsk_rate','dl01_tokenword_oldhsk_rate','dl01_tokenword_hgj_rate','dl01_tokenchar_oldhsk_rate','dl03_tokenchar_hgj_rate','dl03_tokenchar_oldhsk_rate','oov_tokenword_hgj_rate','var_dl_tokenchar_hgj_rate','oov_typeword_oldhsk','dl01_typeword_hgj','oov_typeword_huayu','dl02_typeword_newhsk','dl05_typechar_newhsk','dl01_typeword_newhsk','dl04_typechar_oldhsk','dl06_typechar_newhsk','oov_typechar_oldhsk','dl01_typeword_oldhsk','oov_typechar_newhsk','dl02_typechar_oldhsk','dl03_typechar_hgj','oov_typeword_newhsk','dl01_typeword_huayu','oov_typeword_hgj','dl01_typechar_newhsk','oov_typechar_hgj','dl02_typechar_hgj','dl03_typechar_oldhsk','dl05_typeword_hgj','max_modilen', 'argv_modilen', 'argv_symno', 'max_symno', 'min_symno', 'var_symno','PronDemo_rate', 'PronPers_Firs_rate', 'PronPers_Secd_rate', 'PronPers_Thir_rate', 'PronWhat_rate', 'binlieConj_rate', 'dijinConj_rate', 'jissheConj_rate', 'rangbuConj_rate', 'shunchengConj_rate', 'tiaojianConj_rate', 'yinguoConj_rate', 'zhuanzheConj_rate', 'ConjNeg_rate', 'ConjPosi_rate', 'ConjAll_rate','once_word_rate', 'entity_rate', 'neg_rate', 'cul_word_rate', 'cont_rate', 'len1_rate', 'len2_rate', 'len3_rate', 'len4_rate', 'len4__rate', 'lenworddie4_rate', 'lenworddie4__rate', 'lenworddie3_rate', 'lenworddie3__rate', 'lenworddie2_rate','argv_bh', 'low8_bh_rate', 'argv_corf_type_char', 'argv_logf_type_char', 'argv_logf_token_char', 'argv_corrank_type_char', 'argv_corrank_token_char', 'common_10_rate', 'common_20_rate', 'oov_corf1000_char_rate', 'oov_corf1500_char_rate', 'oov_corf2000_char_rate', 'oov_corf500_char_rate', 'argv_subf_type_char', 'argv_sublogf_token_char', 'argv_sublogf_type_char', 'argv_subrank_token_char', 'max_subrank_token_char', 'argv_subrank_type_char', 'max_subrank_type_char', 'argv_sublogrank_token_char', 'max_sublogrank_token_char', 'argv_sublogrank_type_char', 'max_sublogrank_type_char', 'oov_corf1000_word_rate', 'oov_corf2000_word_rate', 'oov_corf1500_word_rate', 'oov_corf500_word_rate', 'argv_corrank_type_word', 'argv_corrank_token_word', 'argv_corf_type_word', 'argv_logf_type_word', 'oov_corf_word', 'r_rate', 'z_rate', 'PronAll_rate', 'PronPers_rate', 'diffwordtype_1500_freq', 'diffwordtype_3000_freq', 'diffwordtype_1500_sub', 'diffwordtype_3000_sub', 'diffwordtype_1500_contemp', 'diffwordtype_3000_contemp', 'argv_cul', 'max_cul', 'var_cul', 'sum_mood_q', 'sum_mood_j', 'argv_alleduwords_level', 'argv_wencai_score', 'max_wencai_score', 'var_wencai_score', 'sum_wencai_score', 'lowmeaning_rate', 'wc_z_rate', 'wc_f_rate','highmeaning_rate','argv_bh1','maxi_bh1','newmod','sense_sc','dic0','pro']].values
from sklearn.preprocessing import MinMaxScaler
min_max_sacler=MinMaxScaler()

test=min_max_sacler.fit_transform(X)
print(test)

print('预测结果：\n',(SVM.predict(test))[-1])




