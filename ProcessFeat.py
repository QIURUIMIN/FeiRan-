# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:14:27 2019

@author: sydney
"""

import re,os
import pandas as pd 
import numpy as np

from zhon.hanzi import punctuation
from zhon.hanzi import characters
'''
'''
featdata_name='model\\output\\test.csv'

def deal(featdata_name):
	# featdata_name='code/test.csv'

	root = os.path.abspath(os.path.join(os.getcwd(),'..'))
	feat_path = os.path.join(root,featdata_name)

	data = pd.read_csv(feat_path,encoding='utf8',sep=',')
	print(data)
	print('basic info:','\n',data.info())


	#总字数、总句数
	data['zi_amount']  = data.apply(lambda x: x['mean_lenz'] * x['sent'], axis=1)
	data['ci_amount']  = data.apply(lambda x: x['mean_lenw'] * x['sent'], axis=1)
	#中笔画数
	data['mid_bh'] = data.apply(lambda x: x['zi_amount'] - x['low8_bh'] - x['hi16_bh'], axis=1)
	#类符字词比
	data['charword_typo']  = data.apply(lambda x: x['typo_char']/x['typo_word'], axis=1)
	#所有名词
	data['all_n'] = data.apply(lambda x: x['n']+x['nd']+x['nh']+x['ni']+x['nl']+x['ns']+x['nt']+x['nz'], axis=1)
	#所有形容词
	data['all_adj'] = data.apply(lambda x: x['a']+x['b'], axis=1)
	data['wc_z']=data.apply(lambda x: x['B']+x['C']+x['E']+x['J']+x['Ac']+x['Be']+x['Bf']+x['Bg']+x['Bh']+x['Bo']+x['Ca']+x['Cb']+x['Da']+x['Dc']+x['Df']+x['Ea']+x['Eb']+x['Ec']+x['Ef']+x['Ga']+x['Ia']+x['Ib']+x['Jb'], axis=1)
	data['wc_f']=data.apply(lambda x: x['A']+x['D']+x['H']+x['K']+x['Aa']+x['Ab']+x['Ae']+x['Ah']+x['Al']+x['Ba']+x['Di']+x['Dj']+x['Dm']+x['Fc']+x['Gc']+x['Hc']+x['Hg']+x['Hi']+x['Ig']+x['Jc']+x['Ka']+x['Kd']+x['Ke'], axis=1)
	data['argv_bh1']=data.apply(lambda x: x['argv_bh']+x['mid_bh']/x['zi_amount'], axis=1)
	data['maxi_bh1']=data.apply(lambda x: x['maxi_bh']+x['hi16_bh']/x['zi_amount'], axis=1)

	data['newmod']=data.apply(lambda x: (x['n']+x['nd']+x['nh']+x['ni']+x['nl']+x['ns']+x['nt']+x['nz']+x['a']+x['b']+x['idiom']+x['yc']+x['lianmian']+x['dg_word']+x['gy_word'])/x['ci_amount']+x['MVR'], axis=1)
	# data['newmod']=data.apply(lambda x: x['newmod']/x['ci_amount']+x['MVR'], axis=1)

	data['sense_sc']=data.apply(lambda x: x['sensetouchword']+x['sensesightword']+x['sensehearingword']+x['sensesmellword']+x['sensetasteword']+x['sc_word'], axis=1)
	data['dic0']=data.apply(lambda x: x['fy_word']+x['guanyong_word']+x['suolue_word']+x['kouyu_word']+x['ci_classsyn1']+x['ci_classsyn2'], axis=1)

	data['pro']=data.apply(lambda x: x['PronPers']+x['r'], axis=1)

	#除以字数
	'''
	low8_bh、mid_bh、hi16_bh、symm_char、once_char
	oov_corf_char、oov_corf500_char、oov_corf1000_char、oov_corf1500_char、oov_corf2000_char、oov_commonchar
	common_10、common_20、common_35、
	'''
	minus_zi_lis = ['low8_bh','mid_bh','hi16_bh','symm_char','once_char','oov_corf_char','oov_corf500_char','oov_corf1000_char','oov_corf1500_char','oov_corf2000_char','oov_commonchar','common_10','common_20','common_35']
	for l in minus_zi_lis:
		new_name = l+'_rate'
		# print(new_name)
		data[new_name] = data.apply(lambda x: x[l]/x['zi_amount'], axis=1)

	#除以词数
	'''
	once_word,
	oov_corf_word,oov_corf500_word,oov_corf1000_word,oov_corf1500_word,oov_corf2000_word,
	len1、len2、len3、len4、len4_、lenworddie4、lenworddie4_,lenworddie3,lenworddie3_,lenworddie2
	 entity, neg, idiom, yc,lianmian,sensetouchword, sensespaceword, sensesightword, sensehearingword, sensesmellword, sensetasteword,
		allsensewords,dg_word, gy_word, sc_word, fy_word, guanyong_word, suolue_word, kouyu_word,cul_word, ci_classsyn1, ci_classsyn2,
		wc_z,wc_f,
		highmeaning,
		lowmeaning,
	cont,func,a,b,c,d,e,g,h,i,j,k,m,n,nd,nh,ni,nl,ns,nt,nz,o,p,q,r,u,v,wp,ws,x,z,
		PronAll,PronDemo,PronPers,PronPers_Firs,PronPers_Secd,PronPers_Thir,PronWhat,
		binlieConj,dijinConj,jissheConj,rangbuConj,shunchengConj,tiaojianConj,yinguoConj,zhuanzheConj,ConjNeg,ConjPosi,ConjAll
	'''
	miuns_ci_lis = ['argv_dl_tokenchar_oldhsk', 'dl01_tokenword_huayu', 'var_dl_tokenchar_oldhsk', 'dl02_tokenchar_newhsk', 'oov_tokenword_oldhsk', 'dl07_tokenword_huayu', 'dl06_tokenchar_newhsk', 'dl01_tokenword_newhsk', 'dl04_tokenchar_hgj', 'argv_dl_tokenword_hgj', 'dl04_tokenchar_newhsk', 'dl03_tokenword_huayu', 'dl04_tokenword_hgj', 'var_dl_tokenword_newhsk', 'var_dl_tokenchar_newhsk', 'dl03_tokenword_hgj', 'argv_dl_tokenword_oldhsk', 'var_dl_tokenchar_hgj', 'dl02_tokenword_oldhsk', 'dl03_tokenchar_oldhsk', 'dl03_tokenchar_newhsk', 'dl05_tokenword_huayu', 'argv_dl_tokenword_huayu', 'dl06_tokenword_huayu', 'dl04_tokenword_huayu', 'dl02_tokenword_huayu', 'oov_tokenword_hgj', 'dl01_tokenchar_oldhsk', 'dl01_tokenchar_newhsk', 'dl05_tokenword_newhsk', 'dl04_tokenword_oldhsk', 'dl01_tokenword_oldhsk', 'oov_tokenword_huayu', 'var_dl_tokenword_hgj', 'dl05_tokenword_hgj', 'dl01_tokenword_hgj', 'dl03_tokenword_oldhsk', 'dl02_tokenword_hgj', 'argv_dl_tokenchar_hgj', 'dl02_tokenchar_hgj', 'dl02_tokenword_newhsk', 'dl04_typeword_oldhsk','oov_tokenchar_hgj', 'dl04_tokenchar_oldhsk', 'oov_tokenchar_oldhsk', 'argv_dl_tokenword_newhsk', 'dl06_tokenword_hgj', 'dl03_tokenchar_hgj', 'dl06_tokenword_newhsk', 'dl04_tokenword_newhsk', 'var_dl_tokenword_huayu', 'dl02_tokenchar_oldhsk', 'dl05_tokenchar_newhsk', 'oov_tokenchar_newhsk', 'dl03_tokenword_newhsk', 'oov_tokenword_newhsk', 'argv_dl_tokenchar_newhsk', 'dl01_tokenchar_hgj', 'var_dl_tokenword_oldhsk','highmeaning','lowmeaning','wc_z','wc_f','all_n','once_word', 'oov_corf_word', 'oov_corf500_word', 'oov_corf1000_word', 'oov_corf1500_word', 'oov_corf2000_word', 'len1', 'len2', 'len3', 'len4', 'len4_', 'lenworddie4', 'lenworddie4_', 'lenworddie3', 'lenworddie3_', 'lenworddie2', 'cont', 'func', 'all_adj', 'entity', 'neg', 'idiom', 'yc', 'lianmian', 'sensetouchword', 'sensespaceword', 'sensesightword', 'sensehearingword', 'sensesmellword', 'sensetasteword', 'allsensewords', 'dg_word', 'gy_word', 'sc_word', 'fy_word', 'guanyong_word', 'suolue_word', 'kouyu_word', 'cul_word', 'ci_classsyn1', 'ci_classsyn2','cont', 'func', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z', 'PronAll', 'PronDemo', 'PronPers', 'PronPers_Firs', 'PronPers_Secd', 'PronPers_Thir', 'PronWhat', 'binlieConj', 'dijinConj', 'jissheConj', 'rangbuConj', 'shunchengConj', 'tiaojianConj', 'yinguoConj', 'zhuanzheConj', 'ConjNeg', 'ConjPosi', 'ConjAll']
	for l in miuns_ci_lis:
		new_name = l+'_rate'
		# print(l)
		# print(new_name)
		data[new_name] = data.apply(lambda x: x[l]/x['ci_amount'], axis=1)

	#add
	list=['sense_sc','dic0','pro']
	for l in list:
		new_name=l
		data[new_name]=data.apply(lambda x: x[l]/x['ci_amount'], axis=1)



	#除以句数
	miuns_ci_lis = ['easy_sent','complex_sent']
	for l in miuns_ci_lis:
		new_name = l+'_rate'
		# print(l)
		# print(new_name)
		data[new_name] = data.apply(lambda x: x[l]/x['sent'], axis=1)


	#final_data
#level,txtname,
	all_featname = """txtname,argv_bh,maxi_bh,bh_freq_corpus,mid_bh_rate,low8_bh_rate,hi16_bh_rate,hi20_bh,symm_char_rate,argv_corf_type_char,argv_corf_token_char,argv_logf_type_char,argv_logf_token_char,argv_corrank_type_char,argv_corrank_token_char,common_10_rate,common_20_rate,common_35_rate,oov_corf_char,oov_corf1000_char_rate,oov_corf1500_char_rate,oov_corf2000_char_rate,oov_corf500_char_rate,typo_char,ttr_char,root_char,log_char,once_char_rate,argv_docfreq_char,maxi_docfreq_char,mini_docfreq_char,var_docfreq_char,argv_docfreq_word,maxi_docfreq_word,mini_docfreq_word,var_docfreq_word,len1_rate,len2_rate,len3_rate,len4_rate,len4__rate,lenworddie4_rate,lenworddie4__rate,lenworddie3_rate,lenworddie3__rate,lenworddie2_rate,mean_wlen_strokes,maxi_strokes,mean_wlen_py,maxi_py,mean_wlen,maxi,charword_token,charword_typo,oov_corf1000_word_rate,oov_corf2000_word_rate,oov_corf1500_word_rate,oov_corf500_word_rate,argv_corrank_type_word,argv_corrank_token_word,argv_corf_type_word,argv_logf_type_word,argv_corf_token_word,argv_logf_token_word,oov_corf_word,typo_word,ttr_word,root_word,log_word,once_word_rate,wc_z_rate,wc_f_rate,highmeaning_rate,lowmeaning_rate,all_n_rate,all_adj_rate,entity_rate,neg_rate,idiom_rate,yc_rate,lianmian_rate,sensetouchword_rate, sensespaceword_rate, sensesightword_rate, sensehearingword_rate, sensesmellword_rate, sensetasteword_rate,allsensewords_rate,dg_word_rate, gy_word_rate, sc_word_rate, fy_word_rate, guanyong_word_rate, suolue_word_rate, kouyu_word_rate,cul_word_rate, ci_classsyn1_rate, ci_classsyn2_rate,cont_rate,func_rate,a_rate,b_rate,c_rate,d_rate,e_rate,g_rate,h_rate,i_rate,j_rate,k_rate,m_rate,n_rate,nd_rate,nh_rate,ni_rate,nl_rate,ns_rate,nt_rate,nz_rate,o_rate,p_rate,q_rate,r_rate,u_rate,v_rate,wp_rate,ws_rate,x_rate,z_rate,PronAll_rate,PronDemo_rate,PronPers_rate,PronPers_Firs_rate,PronPers_Secd_rate,PronPers_Thir_rate,PronWhat_rate,binlieConj_rate,dijinConj_rate,jissheConj_rate,rangbuConj_rate,shunchengConj_rate,tiaojianConj_rate,yinguoConj_rate,zhuanzheConj_rate,ConjNeg_rate,ConjPosi_rate,ConjAll_rate,diffchartoken_1500_freq,diffchartype_1500_freq,diffchartoken_3000_freq,diffchartype_3000_freq,diffchartoken_1500_sub,diffchartype_1500_sub,diffchartoken_3000_sub,diffchartype_3000_sub,diffwordtoken_1500_freq, diffwordtype_1500_freq, diffwordtoken_3000_freq, diffwordtype_3000_freq,diffwordtoken_1500_sub, diffwordtype_1500_sub, diffwordtoken_3000_sub, diffwordtype_3000_sub,diffwordtoken_1500_contemp, diffwordtype_1500_contemp, diffwordtoken_3000_contemp,diffwordtype_3000_contemp,argv_mean, max_mean, min_mean, argv_cul,max_cul, min_cul, var_cul,argv_mood_q, max_mood_q, min_mood_q, var_mood_q,sum_mood_q,argv_mood_j, max_mood_j, min_mood_j, var_mood_j,sum_mood_j,argv_alleduwords_level, max_alleduwords_level, min_alleduwords_level, var_alleduwords_level,argv_conc, max_conc, min_conc, var_conc,argv_symno, max_symno, min_symno, var_symno,argv_wencai_score, max_wencai_score, min_wencai_score, var_wencai_score,sum_wencai_score,SF,easy_sent_rate,complex_sent_rate,mean_posnum,max_posnum,min_posnum,argv_class0,max_class0,min_class0,argv_class1,max_class1,min_class1,argv_class2,max_class2,min_class2, MOD,MVR,a_rank,b_rank,c_rank,d_rank,e_rank,g_rank,h_rank,i_rank,j_rank,k_rank,m_rank,n_rank,nd_rank,nh_rank,ni_rank,nl_rank,ns_rank,nt_rank,nz_rank,o_rank,p_rank,q_rank,r_rank,u_rank,v_rank,wp_rank,ws_rank,x_rank,z_rank,VP_num,NP_num,ADVP_num,PP_num,ADJP_num,VPmax_phlen,NPmax_phlen,ADVPmax_phlen,PPmax_phlen,ADJPmax_phlen,VPmax_phht,NPmax_phht,ADVPmax_phht,PPmax_phht,ADJPmax_phht,VPargv_phlen,NPargv_phlen,ADVPargv_phlen,PPargv_phlen,ADJPargv_phlen,VPargv_phht,NPargv_phht,ADVPargv_phht,PPargv_phht,ADJPargv_phht,mean_lenz,max_lenz,mean_lenw,max_lenw,mean_th,mean_mdd,argv_modilen,mean_mainv,max_mainv,mdd_text,modi_num,max_th,max_mdd,max_modilen,zi_amount,ci_amount,sent,max_lenw_pg,max_lenz_pg,mean_lenw_pg,mean_lenz_pg,argv_dl_tokenword_oldhsk_rate,dl02_tokenchar_hgj_rate,dl04_tokenchar_newhsk_rate,dl03_tokenword_oldhsk_rate,var_dl_tokenchar_oldhsk_rate,dl02_tokenword_newhsk_rate,dl04_tokenchar_hgj_rate,dl03_tokenword_huayu_rate,dl02_tokenword_hgj_rate,dl01_tokenchar_newhsk_rate,dl04_typeword_oldhsk,oov_tokenchar_hgj_rate,var_dl_tokenchar_newhsk_rate,dl04_tokenword_newhsk_rate,dl02_tokenchar_newhsk_rate,oov_tokenword_oldhsk_rate,argv_dl_tokenword_huayu_rate,dl06_tokenchar_newhsk_rate,oov_tokenchar_oldhsk_rate,dl04_tokenword_huayu_rate,dl02_tokenchar_oldhsk_rate,var_dl_tokenword_hgj_rate,dl05_tokenword_hgj_rate,dl01_tokenword_huayu_rate,dl02_tokenword_huayu_rate,var_dl_tokenword_huayu_rate,argv_dl_tokenchar_oldhsk_rate,var_dl_tokenword_oldhsk_rate,argv_dl_tokenchar_hgj_rate,oov_tokenword_newhsk_rate,var_dl_tokenword_newhsk_rate,dl06_tokenword_huayu_rate,dl06_tokenword_hgj_rate,oov_tokenchar_newhsk_rate,dl03_tokenchar_newhsk_rate,dl05_tokenword_huayu_rate,dl01_tokenword_oldhsk_rate,dl01_tokenchar_hgj_rate,dl06_tokenword_newhsk_rate,dl03_tokenword_newhsk_rate,dl07_tokenword_huayu_rate,dl01_tokenword_hgj_rate,dl04_tokenchar_oldhsk_rate,dl01_tokenchar_oldhsk_rate,argv_dl_tokenchar_newhsk_rate,dl03_tokenchar_hgj_rate,argv_dl_tokenword_newhsk_rate,argv_dl_tokenword_hgj_rate,dl04_tokenword_hgj_rate,dl03_tokenchar_oldhsk_rate,dl01_tokenword_newhsk_rate,dl04_tokenword_oldhsk_rate,dl05_tokenword_newhsk_rate,oov_tokenword_hgj_rate,dl02_tokenword_oldhsk_rate,oov_tokenword_huayu_rate,dl03_tokenword_hgj_rate,var_dl_tokenchar_hgj_rate,dl05_tokenchar_newhsk_rate, dl03_typeword_oldhsk,dl04_typeword_hgj,dl03_typeword_huayu,dl02_typeword_huayu,oov_typeword_oldhsk,dl03_typeword_hgj,dl01_typechar_hgj,dl01_typeword_hgj,oov_typeword_huayu,dl04_typeword_huayu,dl05_typeword_newhsk,dl06_typeword_hgj,dl06_typeword_huayu,dl02_typeword_newhsk,dl03_typeword_newhsk,dl05_typechar_newhsk,dl01_typeword_newhsk,dl04_typechar_oldhsk,dl06_typechar_newhsk,dl02_typeword_hgj,oov_typechar_oldhsk,dl01_typeword_oldhsk,oov_typechar_newhsk,dl02_typechar_oldhsk,dl03_typechar_hgj,dl06_typeword_newhsk,oov_typeword_newhsk,dl03_typechar_newhsk,dl01_typeword_huayu,oov_typeword_hgj,dl04_typechar_newhsk,dl02_typeword_oldhsk,dl01_typechar_oldhsk,dl04_typechar_hgj,dl01_typechar_newhsk,dl02_typechar_newhsk,dl07_typeword_huayu,oov_typechar_hgj,dl05_typeword_huayu,dl02_typechar_hgj,dl03_typechar_oldhsk,dl05_typeword_hgj,dl04_typeword_newhsk,newmod,dic0,sense_sc,pro,argv_bh1,maxi_bh1,max_subrank_token_char,max_subrank_type_char,argv_subrank_token_char,argv_sublogrank_type_char,argv_sublogf_token_char,argv_subrank_type_char,argv_subf_type_char,argv_sublogf_type_char,argv_sublogrank_token_char,max_sublogrank_token_char,max_sublogrank_type_char,mean_op_token_pg,mean_op_v_pg,mean_op_n_pg,mean_op_content_pg,max_op_token_pg,max_op_v_pg,max_op_n_pg,max_op_content_pg,pg_num,mean_op_token_s,mean_op_v_s,mean_op_n_s,mean_op_content_s,max_op_token_s,max_op_v_s,max_op_n_s,max_op_content_s,binlieConj_rate,dijinConj_rate,jissheConj_rate,rangbuConj_rate,shunchengConj_rate,tiaojianConj_rate,yinguoConj_rate,zhuanzheConj_rate""".strip()
	all_featlis = all_featname.replace('\n','').replace(' ','').split(',')
	#print(all_featlis)
	all_featlis = [i for i in all_featlis if len(i)>0]
	# all_featlis=set(all_featlis)
	#print(list(data.columns))
	df = data[all_featlis]

	# l=['ArgStrokes','HighStroke','CharType','CharToken','CharTTR','WordToken','WordType','WordTTR','IdiomWord','NounRate','ContentRate','FuncRate','Cont_Func','NegWord','ArgvSentLenW','EasySentRate','HighTreeRate','NP_num','Modi_num']
	# df=df[l]



	# print(df[:10])


	print(df.info())

	#df.to_csv('/home/test/siyuan/Thesis1001/data/0322_Siyuandata_level.csv',index=False,sep=',')
	root = os.path.abspath(os.path.join(os.getcwd(), ".."))
	process_feat_path = os.path.join(root, 'model/output/liyi.csv')
	print(process_feat_path)
	df.to_csv(process_feat_path,index=False,sep=',')

deal(featdata_name)