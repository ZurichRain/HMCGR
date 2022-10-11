xxx_data 是文档级别的数据集 保留xml文档的原始数据状态 只是将数据转化为一种容易处理的形式
数据格式：
{
    text:xxx
    place:[]
    path:[]
    ...
    qslink:[]
    olink:[]
    movelink:[]
}


clear_xxx_data 是文档级别的数据集 并且对文档内容做了清洗（去掉冗余字符） 而且只保留了当前任务所需要的必要信息
数据格式：
{
    text:xxx
    place:[]
    path:[]
    ...
    qslink:[]
    olink:[]
    movelink:[]
}


clear_seq_xxx_data 是处理成seq的数据集
数据格式：
{
    text:xxx
    place:[]
    path:[]
    ...
    qslink:[]
    olink:[]
    movelink:[]
}
