# -*- coding: utf-8 -*-
import tensorflow as tf
'''
COLUMNS = ['age','sex','user_lv_cd',
           'sku_id','model_id','type','cate','brand',
           'comment_num','has_bad_comment','bad_comment_rate',
           'a1','a2','a3']
'''
           
class Embedding(object):
    
    def __init__(self,
                 features,
                 nums_feature,
                 skuid_voc_size,
                 brand_voc_size
                 ):
        
        self.nums_feature = nums_feature
        
        
        with tf.name_scope('Embedding-layer'):
            
            
            def embed_one_hot(wise_list):
                wise_size = int(max(wise_list))
                return tf.concat([
                        tf.zeros([1,wise_size], dtype=tf.float32),
                        tf.one_hot(range(wise_size), wise_size, 1.0, 0.0)
                        ], 0 )
            
            embedding = []
            USER_LV_CD = [0, 1, 2, 3, 4, 5]# 0 is null
            embedding_user_lv_cd = embed_one_hot(USER_LV_CD)
            
            AGE = [0, 1, 2, 3, 4, 5, 6]# 0 is null
            embedding_age = embed_one_hot(AGE)
            
            SEX = [0, 1, 2]#0 is null
            embedding_sex = embed_one_hot(SEX)
            
            skuid_embedding_size = 128#not-null
            
            embedding_skuid = tf.get_variable(shape = [skuid_voc_size, skuid_embedding_size], name = 'Embedding_skuid')
            
            model_id_size = 350 #-1 is null
            model_id_embedding_size = 16
            
            embedding_model_id = tf.get_variable(shape = [model_id_size, model_id_embedding_size],  name = 'Embedding_modelid')
            
            TYPE = [1, 2, 3, 4, 5, 6]#not-null
            embedding_type = embed_one_hot(TYPE)
            
            CATE = [8, 4, 6, 5, 11, 9, 7, 10]#not-null
            embedding_cate = embed_one_hot(CATE)
            
            
            brand_embedding_size = 16 #not- null
            embedding_brand = tf.get_variable(shape = [brand_voc_size, brand_embedding_size],  name = 'Embedding_brand')
            
            comment_num = [0, 1, 2, 3, 4, 5]#not- null
            embedding_cpmment_num = embed_one_hot(comment_num)
            
            has_bad_comment = [0, 1]#not-null
            embedding_has_bad = embed_one_hot(has_bad_comment)
            
            bad_comment_rate_size = 101 #not-null
            bad_comment_embedding_size = 16
            embedding_bad_comment_rate = tf.get_variable(shape = [bad_comment_rate_size, bad_comment_embedding_size],  name = 'Embedding_bad_comment')
            
            a = [0, 1, 2, 3]#not-null a1,a2,a3 use same embedding
            embedding_a = embed_one_hot(a)

            embedding.append(embedding_age)
            embedding.append(embedding_sex)
            embedding.append(embedding_user_lv_cd)
            embedding.append(embedding_skuid)
            embedding.append(embedding_model_id)
            embedding.append(embedding_type)
            embedding.append(embedding_cate)
            embedding.append(embedding_brand)
            embedding.append(embedding_cpmment_num)
            embedding.append(embedding_has_bad)
            embedding.append(embedding_bad_comment_rate)
            for i in range(3):
                embedding.append(embedding_a)
            self.embedding = embedding
            raw_features = tf.split(features, nums_feature, axis = 3)
            embedding_features = []
            for i, one_feature in enumerate(raw_features):
                temp_embed = tf.nn.embedding_lookup(embedding[i], one_feature)
                embedding_features.append(temp_embed)
            
        
        self.embedding_vec = tf.concat(embedding_features, 4)#shape is (batchsize, days, max_time, 1, embeddingsum(221))
        self.embedding_vec = tf.squeeze(self.embedding_vec, [3])
