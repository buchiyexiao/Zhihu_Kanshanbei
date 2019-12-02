## Python大作业（知乎看山杯题目）

- 解题思路

  作业一共可以分成两个部分，分别是数据的处理和模型的分析，分别对应代码first.py和second.py

  - 数据处理

    数据处理方面由于个人也是第一次接触机器学习方面的知识和内容，因此主要参考了2017年看山杯上一些优秀选手的思路和代码

    一共给予了我们三个数据集，分别是用户信息，题目信息以及答题信息，然后每个特征集里有许多特征

    - user_info.txt数据处理

      首先借助函数体对最后两项数据进行处理，即去除其中共同含有的字母部分

      然后经过复杂的浏览数据，其实就是数一共多少行时顺道看了一眼（手动滑稽），发现第六个参数和第七个参数（注册平台和注册类型）好像均为unknown，经检验发现确实无用

      ![image-20191203014223115](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191203014223115.png)

      由于用户信息中部分信息离线程度较大，因此我们引入了LabelEncoder来将离散型的数据转化为0至n-1之间的数据，以便于后续的分类和处理

      由于用户信息中最主要的信息用户感兴趣的话题，因此我们对用户感兴趣的话题进行进一步的处理

      ```python
      member_info['num_atten_topic'] = member_info['topic_attent'].apply(len)
      member_info['num_interest_topic'] = member_info['topic_interest'].apply(len)
      
      def most_interest_topic(d):
          if len(d) == 0:
              return -1
          return list(d.keys())[np.argmax(list(d.values()))]
      
      def get_interest_values(d):
          if len(d) == 0:
              return [0]
          return list(d.values())
      
      member_info['most_interest_topic'] = member_info['topic_interest'].apply(most_interest_topic)
      member_info['most_interest_topic'] = LabelEncoder().fit_transform(member_info['most_interest_topic'])
      member_info['interest_values'] = member_info['topic_interest'].apply(get_interest_values)
      member_info['min_interest_values'] = member_info['interest_values'].apply(np.min)
      member_info['max_interest_values'] = member_info['interest_values'].apply(np.max)
      member_info['mean_interest_values'] = member_info['interest_values'].apply(np.mean)
      member_info['std_interest_values'] = member_info['interest_values'].apply(np.std)
      ```

      同时，对于其它参数构造计数特征：对具有很好区分度的特征进行单特征计数

      ```python
      for feat in [ 'gender', 'freq', 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2']:
          col_name = '{}_count'.format(feat)
          member_info[col_name] = member_info[feat].map(member_info[feat].value_counts().astype(int))
          member_info.loc[member_info[col_name] < 2 , feat] = -1
          member_info[feat] += 1
          member_info[col_name] = member_info[feat].map(member_info[feat].value_counts().astype(int))
          member_info[col_name] = (member_info[col_name] - member_info[col_name].min()) / (member_info[col_name].max() - member_info[col_name].min())
      ```

    - ques_Info.txt数据处理

      与user_infp.txt的数据处理类似，对其标题和描述的两个参数进行额外分析，即产生计数，将invite_time分隔成为hour和day

      ```python
      question_info['num_title_sw'] = question_info['title_sw_series'].apply(len)
      question_info['num_title_w'] = question_info['title_w_series'].apply(len)
      question_info['num_desc_sw'] = question_info['desc_sw_series'].apply(len)
      question_info['num_desc_w'] = question_info['desc_w_series'].apply(len)
      question_info['num_qtopic'] = question_info['topic'].apply(len)
      ```

    - data.txt数据处理

      正常处理，将invite_time分隔成为hour和day

    - 数据交集处理

      因为我们的三个数据中一定会有一部分存在相交，因此额外进行了数据处理，借助了set集合完成了求交集任务

  - 模型分析

    - 数据的分隔，因为不理解老师所表述的8:1:1的数据占比问题，本来以为老师的考点为k折交叉验证，采用4:1的比例进行分配，后来发现无解题意，更改成8:1:1，实现方法为：

      data一共用10000行，取任意取两个1000行为1，剩下8000行为8即可

    - 本代码共使用了四种机器学习模型

      - 借助lightgbm库中LGBMClassifier实现的Catboost模型
      - 融合了线性回归模型LinearRegression和Ridge回归模型的Stacking模型
      - 融合了线性回归模型LinearRegression，Lasso和Ridge回归模型的Stacking模型
      - Xgboost模型

- 预测结果

  - Catboost

    ![image-20191203020649376](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191203020649376.png)

  - Stacking

    ![image-20191203022124583](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191203022124583.png)

  - Xgboost

    ![image-20191203022250837](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191203022250837.png)

  - Stacking2

    ![image-20191203022346937](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191203022346937.png)

    经过计算可以得到，尽管额外融合了Lasso回归模型，但是Stacking模型的效果仍不如boost模型，Catboost模型和Xgboost模型中，Catboost模型更胜一筹

- 存在问题

  由于初步入门，最后得到的auc的值并不理想，并且在部分机器学习模型的使用方面，依赖于已经写好的完全体函数，同时对参数并没有特别大的重视，如果以后还能有此类的机会或者相关的竞赛，也会尽可能不依赖其它工具去完成，但是目前可能存在部分数学基础未学习，从深层次理解机器学习模型原理并不现实，只能机械化从代码端进行浅要的认识

  而改进方面，首先是需要对各个模型的每个参数进行进一步的微调，其次是最开始的8:1:1是否可以借助一些已知的算法进行分类，如K-折交叉验证进行一定程度的更改（停留在想法阶段，还没有付诸行动）