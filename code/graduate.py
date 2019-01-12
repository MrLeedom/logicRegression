import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# def cartesian(arrays,out=None):
#     '''
#     params:
#         array:list of array-like
#         out:ndarray
#     returns:
#         out:ndarray
#     '''
#     arrays = [np.asarray(x) for x in arrays]
#     dtype = arrays[0].dtype
#     n = np.prod([x.size for x in arrays])
#     if out is None:
#         out = np.zeros([n,len(arrays)],dtype=dtype)
#     m = n/arrays[0].size
#     out[:,0] = np.repeat(arrays[0],m)
#     if arrays[1:]:
#         cartesian(arrays[1:],out=out[0:m,1:])
#         for j in range(1,arrays[0].size):
#             out[j*m:(j+1)*m,1:] = out[0:m,1:]
#     return out
# def cartesian(*arrays):
#     #standard numpy meshgrid 
#     mesh = np.meshgrid(*arrays)
#     #number of dimensions
#     dim = len(mesh)
#     #number of elements,any index will do
#     elements = mesh[0].size
#     #flatten the whole meshgrid
#     flat = np.concatenate(mesh).ravel()
#     #reshape and transpose
#     reshape = np.reshape(flat,(dim,elements)).T
#     return reshape
def cartesian(*arrays):
    N = len(arrays)
    return np.transpose(np.meshgrid(*arrays,indexing='ij'),np.roll(np.arange(N+1),-1)).reshape(-1,N)
#加载数据
df = pd.read_csv('../data/binary.csv')

# #浏览数据
# print(df.head())

# #重命名'rank'列，因为dataframe中有个方法名也为'rank'
df.columns=["admit","gre","gpa","prestige"]
# print(df.columns)
# print(df.describe())
# print(df.std())

# #频率表，表示prestige和admin的值相应的数量关系
# #这一块的几个方法值得好好地去学习一下
# print(pd.crosstab(df.admit,df.prestige,rownames=['admit']))
# df.hist()
# pl.show()

#将prestige设为虚拟变量
dummy_ranks=pd.get_dummies(df['prestige'],prefix='prestige')
print(dummy_ranks.head())

#为逻辑回归创建所需的data frame
#除admit,gre,gpa外，加入了上面常见的虚拟变量（注意，引入的虚拟变量列数应为虚拟变量总列数减１，减去的１列　作为基准）
cols_to_keep = ['admit','gre','gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:,'prestige_2':])
print(data.head())
#将新的虚拟变量加入到原始的数据集中，就不再需要原来的prestige列了，在此强调一点，生成m个虚拟变量后，只要加入m-1个
# 虚拟变量到数据集中，未引入的一个是作为基准对比的

#指定作为训练变量的列，不含目标列admit
train_cols = data.columns[1:]

logit = sm.Logit(data['admit'],data[train_cols])
#拟合模型
result = logit.fit()


#使用训练模型预测数据
#构建数据集
#与训练集相似，一般也是通过pd.read_csv读入
#在这边为方便，我们将训练集拷贝一份作为预测集（不包括admit）
import copy
combos = copy.deepcopy(data)

#数据中的列要跟预测时用到的列一致
predict_cols = combos.columns[1:]

#预测集也要添加intercept变量
combos['intercept'] = 1.0

#进行预测，并将与测评分存入predict列中
combos['predict'] = result.predict(combos[predict_cols])

#预测完成后，predict的值介于[0,1]间的概率值
# 我们可以根据需要，提取预测结果
# 例如，嘉定predict>0.5，则表示会被录取
# 在这边我们检验一下上述选取结果的精确度
total = 0
hit = 0
for value in combos.values:
    #预测分数predict，是数据列的最后一列
    predict = value[-1]
    #实际录取结果
    admit = int(value[0])

    #假定预测概率大于０．５则表示预测被录取
    if predict > 0.5:
        total += 1
        #表示预测命中
        if admit == 1:
            hit += 1

#输出结果
print('Total: %d,Hit:%d,Precision:%.2f'%(total,hit,100.0*hit/total))
print(result.summary())
#查看每个系数的置信区间,置信区间可以看出模型系数的健壮性
print(result.conf_int())
#相对危险度，可以知道每个变量每单位的增加＼减少对录取几率的影响
print(np.exp(result.params))
#可以用置信区间来计算系数的影响
params = result.params
conf = result.conf_int()
conf['OR']=params
conf.columns=['2.5%','97.5%','OR']
print(np.exp(conf))

#更深入的挖掘，为了评估我们分类器的效果，我们将使用每个输入值的逻辑组合来重新创建数据集，如此可以得知在不同变量下
# 预测录取的可能性的增加＼减少

#根据最大．最小值生成gre,gpa均匀分布的１０个值，而不是生成所有可能的值
gres = np.linspace(data['gre'].min(),data['gre'].max(),10)
print(gres)
gpas = np.linspace(data['gpa'].min(),data['gpa'].max(),10)
print(gpas)
#枚举所有可能性
combos = pd.DataFrame(cartesian(gres,gpas,[1,2,3,4],[1.]))

#重新创建哑变量
combos.columns=['gre','gpa','prestige','intercept']
dummy_ranks = pd.get_dummies(combos['prestige'],prefix='prestige')
dummy_ranks.columns=['prestige_1','prestige_2','prestige_3','prestige_4']

#只保留用于预测的列
cols_to_keep = ['gre','gpa','prestige','intercept']
combos = combos[cols_to_keep].join(dummy_ranks.ix[:,'prestige_2':])

#使用枚举的数据集来做预测
combos['admit_pred'] = result.predict(combos[train_cols])
print(combos.head())

def isolate_and_plot(variables):
    #isolate gre and class rank
    grouped = pd.pivot_table(combos,values=['admit_pred'],index=[variables,'prestige'],aggfunc=np.mean)
    #分离给定的变量和不同的声望等级＼组合的平均可能性，用来分离声望和其他变量

    #make a plot
    colors = 'rgbyrgby'
    for col in combos.prestige.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0),plt_data['admit_pred'],color=colors[int(col)])
        pl.xlabel(variables)
    pl.ylabel("P(admit==1)")
    pl.legend(['1','2','3','4'],loc='upper left',title='Prestige')
    pl.title("Prob(admit=1) isolating "+variables+" and prestige")
    pl.show()

    
isolate_and_plot('gre')
isolate_and_plot('gpa')
    