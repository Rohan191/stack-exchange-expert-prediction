#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:44:33 2018

@author: rohantondulkar
"""

import numpy as np
import pandas as pd
from dateutil import parser
import time
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import re
    
    
class BestAnswererPredictor:
    """ Model to make real time predictions of best answerer for new questions"""
    
    def __init__(self, train_profile, user_details, user_tag_profile, tags_list, user_text_vec, doc2vec_model, ltr_model ):
        # 2-d numpy array for all questions in the format
        # PostId, QuestionContent, AnswerId, AnswerContent, IsBestAnswer, Score, Tags, AnswererId, AnswerTime, TotalContent
        self.train_profile = train_profile
        
        # Get 2-D array of user details in the format 
        # UserId, AnswerFrequencyInDays, Reputation, NumBestAnsToAnsRatio, MRR, AvgQuesQuality, AvgQuesPopularity
        self.user_details = user_details
        self.user_list = self.user_details[:,0]
        self.num_users = len(self.user_list)
        
        # Get 2-D array for tag scores on 987 tags per user
        self.user_tag_profile = user_tag_profile
        self.tags_list = tags_list
        
        # Get 2-D array of text vector representation per user
        self.user_text_vec = user_text_vec
        
        self.doc2vec_model = doc2vec_model
        self.ltr_model = ltr_model
        self.stop_words = set(stopwords.words('english'))
        self.pstem = PorterStemmer()
        self.num_features = 9
        self.feature_set = [ self.get_tag_similarity, self.get_cosine_similarity, self.get_user_answer_frequency,
                            self.get_number_best_answers_ratio, self.get_user_reputation, self.get_days_to_prev_ans,
                            self.get_user_MRR, self.get_user_avg_ques_quality, self.get_user_avg_ques_popularity
                           ]
        self.setup()
        
    def setup(self):
        """ Prepare internal data for faster prediction """
        print('Setting up...')
        self.user_train_profiles = {}
        for user in self.user_list:
            self.user_train_profiles[user] = self.train_profile[np.where(self.train_profile[:,7]==user)]
        print('Ready to predict!')
        
    def predict(self, test_data, topK):
        """ Expecting a 1-d numpy array as [QuestionContent, Tags, CreateTime (datetime object)]"""
        # Create a feature set for all ques-user pairs
        features = np.zeros((len(self.user_list), self.num_features + 2))
        features[:, 0] = self.user_list
        
        # Expected order of feature set: 
        # ['SimilarityScore', 'CosineSimilarity', 'AnswerFrequencyInDays', 'NumBestAnsToAnsRatio', 
        #  'Reputation', 'DaysToPrevAns', 'MRR', 'AvgQuesQuality', 'AvgQuesPopularity']     
        for i in range(1,self.num_features):
            #start = time.time()
            features[:, i] = self.feature_set[i-1](test_data)
            #print('Time for feature: {0} is {1} secs'.format(self.feature_set[i-1],time.time()-start))
        features[:,10] = self.ltr_model.predict(features[:,1:self.num_features+1])
        ranked_users = features[features[:,10].argsort()[::-1]][:,0]
        #print(ranked_users[:topK])
        return ranked_users.astype(int)[:topK]
        
    def get_cosine_similarity(self, test_data):
        """ Get cosine similarity between doc2vec representations of question and all users """
        text_score = np.zeros(self.num_users)
        text = test_data[0]
        text = re.compile('\w+').findall(text.lower())
        text = [w for w in text if not w in self.stop_words]
        text = [self.pstem.stem(w) for w in text if len(w)>2]
        text_vec = self.doc2vec_model.infer_vector(text)
        count = 0
        
        for user in self.user_list:
            text_score[count] = cosine_similarity(text_vec.reshape(1, -1), self.user_text_vec[self.user_text_vec[:,0]==user][:,1:])
            count+=1
        return text_score
        
    def get_tag_similarity(self, test_data):
        """ Get tag based similarity between tag profiles of question and all users """
        tags = literal_eval(test_data[1])
        ques_tag_vec = np.zeros(len(self.tags_list))
        
        # Make ques tag vector
        for tag in tags:
            i = np.where(self.tags_list == tag)[0][0]
            ques_tag_vec[i]=1
            
        # Calculate tag score for all users
        count = 0
        tag_score = np.zeros(self.num_users)
        for user in self.user_list:
            user_tag_vec = self.user_tag_profile[np.where(self.user_tag_profile[:, 0]==user)][:,1:].ravel()
            match = user_tag_vec[np.logical_and(ques_tag_vec, user_tag_vec)]
            tag_score[count] = np.sum(match) * match.shape[0]
            count+=1
        return tag_score
    
    def get_days_to_prev_ans(self, test_data):
        """ Get the number of days since last answer for each user """
        days_to_prev_ans = np.zeros(self.num_users)
        create_time = parser.parse(test_data[2])
        count = 0
        
        for user in self.user_list:
            # Get the list of answers by the given userId
            ans_profile = self.user_train_profiles[user]

            # Get the days since previous answer
            ans_profile = ans_profile[np.where(ans_profile[:,8]<create_time)]

            if len(ans_profile) == 0:
                days_to_prev_ans[count] = 2000
            else:
                ans_profile = ans_profile[ans_profile[:,8].argsort()[::-1]]
                timeDiff = create_time - ans_profile[0, 8]
                days_to_prev_ans[count] = timeDiff.total_seconds()/(3600*24)        # in days
            count+=1
        return days_to_prev_ans
    
    def get_user_answer_frequency(self, test_data):
        """ Get answering frequeny for all users """
        return self.user_details[:,1]
    
    def get_user_reputation(self, test_data):
        """ Get reputation for all users """
        return self.user_details[:,2]
    
    def get_number_best_answers_ratio(self, test_data):
        """ Get number best answers to number of answers ratio"""
        return self.user_details[:,3]
    
    def get_user_MRR(self, test_data):
        """ Get mean reciprocal rank for every user """
        return self.user_details[:, 4]
    
    def get_user_avg_ques_quality(self, test_data):
        """ Get average question quality per user"""
        return self.user_details[:, 5]
    
    def get_user_avg_ques_popularity(self, test_data):
        """ Get average question popularty per user"""
        return self.user_details[:, 6]
    

def sample_predictions():
    """ An example to show how to make sample predictions"""
    # All features should be only for 1339 final set of users
    # Load all the files needed
    doc2vecModel = Doc2Vec.load('../Models/user_doc2vec_1000.model')
    user_details = pd.read_csv('../Dataset/UserInformation.csv')
    train_profile = pd.read_csv('../Dataset/Train_Profile.csv')
    train_profile['AnswerTime'] = pd.to_datetime(train_profile['AnswerTime'])
    test_ltr = pd.read_csv('../Dataset/Sample_test_data.csv')
    user_tag_profile = pd.read_csv('../Dataset/User_tag_profile.csv')
    user_text_vec = pd.read_csv('../Dataset/User_text_vec.csv')

    with open(r"../Models/LTR_ALL_final.pkl", "rb") as input_file:
        ltr_model = pickle.load(input_file)
        
    # Initialize the model
    predictor = BestAnswererPredictor(train_profile = train_profile.values, user_details = user_details.values, \
                                  user_tag_profile = user_tag_profile.values, tags_list = user_tag_profile.columns[1:],\
                                  user_text_vec = user_text_vec.values,\
                                  doc2vec_model = doc2vecModel, ltr_model = ltr_model)
    topK = 10
    for index, row in test_ltr.iterrows():
        start = time.time()
        topK_users = predictor.predict(row.values, topK)
        print('\nPrediction took: {0} secs'.format(time.time()-start))
        print('Top {0} users: {1}'.format(topK, topK_users))
        
if __name__ == "__main__":
    sample_predictions
    