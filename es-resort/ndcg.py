import numpy as np
import collections
# from sklearn.metrics import ndcg_score


def validate(qs, targets, preds, k):
    """
    Predicts the scores for the test dataset and calculates the NDCG value.
    Parameters
    ----------
    data : Numpy array of documents
        Numpy array of documents with each document's format is [relevance score, query index, feature vector]
    k : int
        this is used to compute the NDCG@k

    Returns
    -------
    average_ndcg : float
        This is the average NDCG value of all the queries
    predicted_scores : Numpy array of scores
        This contains an array or the predicted scores for the documents.
    """
    # query_groups = get_groups2(qs)  # (group,from,to),一个元组,表示这个qid的样本从哪到哪 以group为一组
    query_groups = get_groups(qs)  # (qid,from,to),一个元组,表示这个qid的样本从哪到哪 以相同query为一组
    all_ndcg = []
    every_qid_ndcg = collections.OrderedDict()

    for qid, a, b in query_groups:
        predicted_sorted_indexes = np.argsort(preds[a:b])[::-1]  # 从大到小的索引
        t_results = targets[a:b]  # 目标数据的相关度label
        t_results = t_results[predicted_sorted_indexes]  # 是按预测概率重新排序的目标数据的相关度label
        # 改进
        t_results = [((x+1)/3) for x in t_results]    # 0.333../0.666../1 ---标签+1
        if set(t_results)=={0.3333333333333333} or len(set(t_results))==1:    # 抛弃所有无点击label（即全为0）的group
            continue
        
        # # Ranking for predicted Value
        # predicted_ranking_list = list(range(len(preds[a:b])))
        # # predicted_ranking_list = sorted(range(len(preds[a:b])), key=lambda k: preds[a:b][k])    # .all()
        # predicted_ranking_list.reverse()  # 从大到小的索引 [::-1]

        # # Ranking for Ideal Document Relevance
        # ideal_ranking_list = sorted(
        #     range(len(targets[a:b])), key=lambda k: targets[a:b][k]
        # )
        # ideal_ranking_list.reverse()

        # dcg_val = dcg_k(targets[a:b][np.array(predicted_ranking_list)], k)
        # idcg_val = dcg_k(targets[a:b][np.array(ideal_ranking_list)], k)
        dcg_val = dcg_k(t_results, k)
        idcg_val = dcg_k([score for score in sorted(t_results)[::-1]], k)
        ndcg_val = dcg_val / idcg_val
        all_ndcg.append(ndcg_val)
        every_qid_ndcg.setdefault(qid, ndcg_val)
        
        # print("predicted_sorted_indexes", predicted_sorted_indexes)
        # print("predict: ", [score for score in (t_results)])
        # print("label  : ", targets[a:b])
        # print("ndcg   : ", ndcg_val)

    average_ndcg = np.nanmean(all_ndcg)
    return average_ndcg, every_qid_ndcg

    """
    for query in query_indexes:
        results = np.zeros(len(query_indexes[query]))

        for tree in self.trees:
            results += self.learning_rate * tree.predict(data[query_indexes[query], 2:])
        predicted_sorted_indexes = np.argsort(results)[::-1]
        t_results = data[query_indexes[query], 0] # 第0列的相关度
        t_results = t_results[predicted_sorted_indexes]

        dcg_val = dcg_k(t_results, k)
        idcg_val = ideal_dcg_k(t_results, k)
        ndcg_val = (dcg_val / idcg_val)
        average_ndcg.append(ndcg_val)
    average_ndcg = np.nanmean(average_ndcg)
    return average_ndcg
"""


def get_groups(qids):
    """Makes an iterator of query groups on the provided list of query ids.

    Parameters
    ----------
    qids : array_like of shape = [n_samples]
        List of query ids.

    Yields
    ------
    row : (qid, int, int)
        Tuple of query id, from, to.
        ``[i for i, q in enumerate(qids) if q == qid] == range(from, to)``

    """
    prev_qid = None
    prev_limit = 0
    total = 0

    for i, qid in enumerate(qids):
        total += 1
        if qid != prev_qid:
            if i != prev_limit:
                yield (prev_qid, prev_limit, i)
            prev_qid = qid
            prev_limit = i

    if prev_limit != total:
        yield (prev_qid, prev_limit, total)


def get_groups2(q_test):
    prev = 0
    for i, group in enumerate(q_test):
        group = int(group)
        # print(i, prev, prev + group)
        yield (i, prev, prev + group)
        prev = prev + group


def group_queries(training_data, qid_index):
    """
        Returns a dictionary that groups the documents by their query ids.
        Parameters
        ----------
        training_data : Numpy array of lists
            Contains a list of document information. Each document's format is [relevance score, query index, feature vector]
        qid_index : int
            This is the index where the qid is located in the training data

        Returns
        -------
        query_indexes : dictionary
            The keys were the different query ids and teh values were the indexes in the training data that are associated of those keys.
    """
    query_indexes = (
        {}
    )  # 每个qid对应的样本索引范围,比如qid=1020,那么此qid在training data中的训练样本从0到100的范围, { key=str,value=[] }
    index = 0
    for record in training_data:
        query_indexes.setdefault(record[qid_index], [])
        query_indexes[record[qid_index]].append(index)
        index += 1
    return query_indexes


def dcg_k(scores, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values. 
    
    Example
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    return np.sum(
        [(np.power(2, scores[i]) - 1) / np.log2(i + 2) for i in range(len(scores[:k]))]
    )  # [0,1] 
    # return scores[0] + np.sum(scores[1:] / np.log2(np.arange(2, len(scores) + 1)))
    # return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))


def ideal_dcg_k(scores, k):
    """
    前k个理想状态下的dcg
        Returns the Ideal DCG value of the list of scores and truncates to k values.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    # 相关度降序排序
    # scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)


# x = validate([1,1,1,1,1,1,1,1,1], np.array([0.6666666666666666, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]), [0.3333333333333333, 0.3333333333333333, 0.6666666666666666, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 80)
# print(x)
# x = validate([1,1,1,1,1,1,1,1,1], np.array([1,0,0,0,0,0,0,0,0,0]), [0,0,1,0,0,0,0,0,0,0], 80)
# print(x)