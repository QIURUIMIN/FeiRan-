1.文件说明
corpus——放入待处理的文本
output——存放提取的特征，里面放了一个已提取好的文件。
parser_ly——语料预处理
featanalyzall、feature、utils——特征提取工具
processfeat——特征加工
wencai_model——存储的svm训练结果
resource——分析以及提取特征时所需要的外部资源

test_model——总运行程序

注意：
注意路径问题，每个程序里其实已经有一段注释掉的读取路径的代码，修改一下，可以运行的更顺畅。

2.corpus——输入的文本格式：
##题目
（空行）
文本内容段1。
（空行）
文本内容段2。


注意：
1.一定要有文本名称
2.文本名称与文本内容之间空一行
3.文本文件为utf-8编码
4.文本内容中，段与段中间同样也是有个空行。
示例：
##sw0.2level0

在渤海新区成为一颗耀眼的新星，让敬业的名字飘进了千家万户。他一生的选择就是认识到了他挚爱的敬业，他一生的热情就是投入到他勤奋的钻研中，他一生的成绩都是敬业发展史上的上的一段佳话，我们怀念敬业的模范干部李厂长。

正如臧克家在纪念鲁迅先生逝世十三周年时所说过：“有些人活着，他已经死了，有些人死了，他还活着”李厂长虽然已逝，但是他会一直活在我们每一个敬业人的心中。

3.特征说明

feature中包含的特征：
['TXT','Strokes','Sym_once','CharFreq','DocFreq','OOV','DiffLevelChar','DiffLevelWord','DiffChar','DiffWord','CommonChar','TTR','WordLen','WordFreq','WordOther','synmeaning','wcmeaning','SentComplex','WordPos','Phrase','SynTree','Dependency','PgComplex','Overlap','wencaidic']
   
svm中所用到的有效特征：
['dl02_tokenchar_hgj_rate','var_dl_tokenchar_oldhsk_rate','dl01_tokenchar_newhsk_rate','dl04_typeword_oldhsk','oov_tokenchar_hgj_rate','var_dl_tokenchar_newhsk_rate','oov_tokenword_oldhsk_rate','dl06_tokenchar_newhsk_rate','oov_tokenchar_oldhsk_rate','dl02_tokenchar_oldhsk_rate','var_dl_tokenword_hgj_rate','dl05_tokenword_hgj_rate','oov_tokenword_newhsk_rate','oov_tokenchar_newhsk_rate','dl01_tokenword_oldhsk_rate','dl01_tokenword_hgj_rate','dl01_tokenchar_oldhsk_rate','dl03_tokenchar_hgj_rate','dl03_tokenchar_oldhsk_rate','oov_tokenword_hgj_rate','var_dl_tokenchar_hgj_rate','oov_typeword_oldhsk','dl01_typeword_hgj','oov_typeword_huayu','dl02_typeword_newhsk','dl05_typechar_newhsk','dl01_typeword_newhsk','dl04_typechar_oldhsk','dl06_typechar_newhsk','oov_typechar_oldhsk','dl01_typeword_oldhsk','oov_typechar_newhsk','dl02_typechar_oldhsk','dl03_typechar_hgj','oov_typeword_newhsk','dl01_typeword_huayu','oov_typeword_hgj','dl01_typechar_newhsk','oov_typechar_hgj','dl02_typechar_hgj','dl03_typechar_oldhsk','dl05_typeword_hgj','max_modilen', 'argv_modilen', 'argv_symno', 'max_symno', 'min_symno', 'var_symno','PronDemo_rate', 'PronPers_Firs_rate', 'PronPers_Secd_rate', 'PronPers_Thir_rate', 'PronWhat_rate', 'binlieConj_rate', 'dijinConj_rate', 'jissheConj_rate', 'rangbuConj_rate', 'shunchengConj_rate', 'tiaojianConj_rate', 'yinguoConj_rate', 'zhuanzheConj_rate', 'ConjNeg_rate', 'ConjPosi_rate', 'ConjAll_rate','once_word_rate', 'entity_rate', 'neg_rate', 'cul_word_rate', 'cont_rate', 'len1_rate', 'len2_rate', 'len3_rate', 'len4_rate', 'len4__rate', 'lenworddie4_rate', 'lenworddie4__rate', 'lenworddie3_rate', 'lenworddie3__rate', 'lenworddie2_rate','argv_bh', 'low8_bh_rate', 'argv_corf_type_char', 'argv_logf_type_char', 'argv_logf_token_char', 'argv_corrank_type_char', 'argv_corrank_token_char', 'common_10_rate', 'common_20_rate', 'oov_corf1000_char_rate', 'oov_corf1500_char_rate', 'oov_corf2000_char_rate', 'oov_corf500_char_rate', 'argv_subf_type_char', 'argv_sublogf_token_char', 'argv_sublogf_type_char', 'argv_subrank_token_char', 'max_subrank_token_char', 'argv_subrank_type_char', 'max_subrank_type_char', 'argv_sublogrank_token_char', 'max_sublogrank_token_char', 'argv_sublogrank_type_char', 'max_sublogrank_type_char', 'oov_corf1000_word_rate', 'oov_corf2000_word_rate', 'oov_corf1500_word_rate', 'oov_corf500_word_rate', 'argv_corrank_type_word', 'argv_corrank_token_word', 'argv_corf_type_word', 'argv_logf_type_word', 'oov_corf_word', 'r_rate', 'z_rate', 'PronAll_rate', 'PronPers_rate', 'diffwordtype_1500_freq', 'diffwordtype_3000_freq', 'diffwordtype_1500_sub', 'diffwordtype_3000_sub', 'diffwordtype_1500_contemp', 'diffwordtype_3000_contemp', 'argv_cul', 'max_cul', 'var_cul', 'sum_mood_q', 'sum_mood_j', 'argv_alleduwords_level', 'argv_wencai_score', 'max_wencai_score', 'var_wencai_score', 'sum_wencai_score', 'lowmeaning_rate', 'wc_z_rate', 'wc_f_rate','highmeaning_rate','argv_bh1','maxi_bh1','newmod','sense_sc','dic0','pro','dic1']