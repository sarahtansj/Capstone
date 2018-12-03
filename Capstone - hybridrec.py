import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate


class hybrid(object):
    
    def __init__ (self,user_id,ratings):
        
        self.user_id = user_id
        self.md = pd.read_csv('./subset.csv')
        self.ratings = ratings
        #print(ratings[(ratings['Cust_Id'] == user_id)][['Cust_Id','Movie_Id', 'Rating']])



        self.collaborative_rating = self.collaborative(self.ratings, self.user_id)
        self.content_rating = self.content_based(self.md,self.ratings,self.user_id)
        self.final_hybrid(self.md, self.collaborative_rating, self.content_rating, self.user_id)
        

    ### Collaborative ##

    def collaborative(self,ratings,user_id):

        reader = Reader()
        #ratings.head()

        temp_ratings = ratings



        data = Dataset.load_from_df(temp_ratings[['Cust_Id','Movie_Id', 'Rating']], reader)
        data.split(n_folds=2)

        ## Training the data ##
        svd = SVD()
        evaluate(svd, data, measures=['RMSE', 'MAE'])

        trainset = data.build_full_trainset()

        algo = SVD()
        algo.fit(trainset)

        #svd.train(trainset)
        ## Testing the data ##

        from collections import defaultdict
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)

        count = 0
     
        for uid, iid, true_r, est, _ in predictions:

             if uid == user_id:
                count = count+1
                temp_ratings.loc[len(temp_ratings)+1]= [uid,iid,est]

        #print("count\n")
        #print(count)
        #print("\n--------here-------\n")
        #print(temp_ratings)

        cb = temp_ratings[(temp_ratings['Cust_Id'] == user_id)][['Movie_Id', 'Rating']]
        #print("\n--------here-------\n")
        #print(cb)
        
        cb = temp_ratings[(temp_ratings['Cust_Id'] == user_id)][['Movie_Id', 'Rating']]

        return(cb)


    ##### CONTENT ######

    def content_based(self,md,ratings,user_id):       

        md['genres']=md['genres'].str.split(';')
        #print(md['Genres'])

        md['soup'] = md['actor'] + md['genres'] + md['actress']
        #print(md['soup'])

        md['soup'] = md['soup'].str.join(' ')

        #md['soup'].fillna({})
        #print(md['soup'])

        count = CountVectorizer(analyzer='word',ngram_range=(1,1),min_df=0, stop_words='english')
        count_matrix = count.fit_transform(md['soup'])
        print (count_matrix.shape)
        #print np.array(count.get_feature_names())
        #print(count_matrix.shape)


        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        def build_user_profiles():
            user_profiles=np.zeros((60001,999))
        #taking only the first 100000 ratings to build user_profile
            for i in range(0,100000):
                
                u=ratings.iloc[i]['Cust_Id']
                b=ratings.iloc[i]['Movie_Id']

                user_profiles[u][b-1]=ratings.iloc[i]['Rating']
                
            return user_profiles

        user_profiles=build_user_profiles()

        def _get_similar_items_to_user_profile(person_id):
            #Computes the cosine similarity between the user profile and all item profiles

            user_ratings = np.empty((999,1))
            cnt=0
            for i in range(0,998):
                movie_sim=cosine_sim[i]
                user_sim=user_profiles[person_id]
                user_ratings[i]=(movie_sim.dot(user_sim))/sum(cosine_sim[i])
            maxval = max(user_ratings)
            print(maxval)

            for i in range(0,998):
                user_ratings[i]=((user_ratings[i]*5.0)/(maxval))

                if(user_ratings[i]>3):

                    cnt+=1

            return user_ratings

        content_ratings = _get_similar_items_to_user_profile(user_id)



        num = md[['Movie_Id']]
        num1 = pd.DataFrame(data=content_ratings[0:,0:])
        frames = [num, num1]


        content_rating = pd.concat(frames, axis =1,join_axes=[num.index])
        content_rating.columns=['Movie_Id', 'content_rating']
        #print(content_rating.shape)
        #print(content_rating)

        return(content_rating)

    
    def final_hybrid(self,md, popularity_rating , collaborative_rating, content_rating, user_id):

        hyb = md[['Movie_Id']]
        title = md[['Movie_Id','Title', 'genres']]

        hyb = hyb.merge(title,on = 'Movie_Id')
        hyb = hyb.merge(self.collaborative_rating,on = 'Movie_Id')
        hyb = hyb.merge(self.content_rating, on='Movie_Id')

        def weighted_rating(x):
            v = x['rating']
            c = x['content_rating']
            return 0.5*v + 0.5 * c


        hyb['hyb_rating'] = hyb.apply(weighted_rating, axis=1)
        hyb = hyb.sort_values('hyb_rating', ascending=False).head(999)
        hyb.columns = ['Movie Id' , 'Title', 'Genres', 'Collaborative Rating', 'Content Rating', 'Hybrid Rating']

        print(len(hyb['Hybrid Rating']))
        print(hyb)