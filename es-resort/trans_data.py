import sys
import re
import codecs 

"""
Input: Ranksvm Format
Output: Feature File / Group File

PS D:\ltr-es\sample\data> python ../trans_data.py train.txt mq2008.train mq2008.train.group
PS D:\ltr-es\sample\data> python ../trans_data.py test.txt mq2008.test mq2008.test.group
PS D:\ltr-es\sample\data> python ../trans_data.py vali.txt mq2008.vali mq2008.vali.group
"""


def save_data(group_data, output_feature, output_group):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data))+"\n")
    for data in group_data:
        # only include nonzero features
        feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]        
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print ("Usage: python trans_data.py [Ranksvm Format Input] [Output Feature File] [Output Group File]")
        sys.exit(0)

    fi = codecs.open(sys.argv[1], encoding='utf-8')
    output_feature = open(sys.argv[2],"w")
    output_group = open(sys.argv[3],"w")
    
    # group_data = []
    # group = ""
    # for line in fi:
    #     if not line:
    #         break
    #     if "#" in line:
    #         line = line[:line.index("#")]
    #     # print(line)
    #     splits = line.strip().split(" ")
    #     if splits[1] != group:
    #         save_data(group_data,output_feature,output_group)
    #         group_data = []
    #     group = splits[1]
    #     group_data.append(splits)
    
    group_data = []
    qid_search = []
    group = ''
    count = 0
    for line in fi:
        if not line:
            break
        if "#" in line:
            comment = line[line.index("#"):].strip('\n').split(' ')
            line = line[:line.index("#")]
        # 通过 input+botCode 来区分一次查询  +opeTime
        splits = line.strip().split(" ")
        e = splits[1] + re.sub("\'", "", ' '.join(comment[-1:]))
        if splits[1] != group or e not in qid_search:
        # if splits[1] != group:
            save_data(group_data, output_feature, output_group)
            group_data = []
            qid_search = []
        qid_search.append(e)
        group = splits[1]   # qid
        group_data.append(splits)

    save_data(group_data,output_feature,output_group)

    fi.close()
    output_feature.close()
    output_group.close()