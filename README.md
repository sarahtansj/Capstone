# Capstone
In this notebook, 4 different recommendation engines have been built based on different ideas and algorithms. They are as follows:

1. Simple Recommender: This system used overall average ratings to build Top Movies Charts, in general and for a specific genre. 
2. Content Based Recommender: I built two content based engines; one that took movie overview and taglines as input and the other which took metadata such as cast, crew, genre and keywords to come up with predictions. I also deviced a simple filter to give greater preference to movies with more votes and higher ratings.
3. Collaborative Filtering: I used the powerful Surprise Library to build a collaborative filter based on single value decomposition. The RMSE obtained was less than 1 and the engine gave estimated ratings for a given user and movie.
4. Hybrid Engine: Bringing together ideas from content and collaborative filterting to build an engine that takes into account both the user's profile and the genres of the movies.
