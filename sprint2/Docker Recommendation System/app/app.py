from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd 
import pickle 
import numpy as np 
import uvicorn 
import random 

import os
print(f'{os.getcwd()},\n going inside app now')
os.chdir('\app')
print(f'{os.getcwd()},\n')
movies_df = pd.read_csv('movies_metadata.csv', low_memory=False)

cosine_sim = None 
indices_cont = None 

with open("cosine_sim.pkl", "rb") as f: 
    print('Loading cosine similarity pickle')
    cosine_sim = pickle.load(f) 
    
    print('Loaded cosine similarity pickle')

with open("indices.pkl", "rb") as f: 
    indices_cont = pickle.load(f) 


with open ("svd.pkl", "rb") as f:
    print("SVD Pickle File Loading")
    svd = pickle.load(f)
    print("SVD Pickle File Loaded!!")

def content_recommender(title, cosine_sim=cosine_sim, df=movies_df,indices=indices_cont, top_n =100):
    print(f"Recommendation for {title} content wise called")
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    try:
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    except:
        pass         
    # top 10 movies 
    sim_scores = sim_scores[1:top_n]
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices].tolist()

def collaborative_recommend(user_id, movies_df, top_n = 10): 
    print(f"Recommnded called for user {user_id}")
    top_movies = [] 
    movies = random.sample(movies_df['title'].tolist(), 200)  
    for movie in movies:
        idx = movies_df.index[movies_df['title'] == movie].tolist()[0]
        score = svd.predict(user_id, idx).est 
        top_movies.append((score, movies_df['title'].iloc[idx]))
    top_movies.sort(reverse=True)
    return [m[1] for m in top_movies][:top_n] 

def hybrid_recommend(user_id, title,svd=svd, cosine_sim=cosine_sim, movies_df=movies_df, indices_cont=indices_cont, top_n = 10): 
    
    # using content based rec
    top_movies_titles  = content_recommender(title, cosine_sim, movies_df, indices_cont, 100)
    
    top_movies = []
    
    for i in top_movies_titles: 
        idx = movies_df.index[movies_df['title'] == i].tolist()[0]
        
        score = svd.predict(user_id, idx).est
        top_movies.append((score, movies_df['title'].iloc[idx]))
    
    # Sort the recommendations based on SVD scores
    top_movies.sort(reverse=True)
    
    return [m[1] for m in top_movies][:top_n] 

app = FastAPI() 

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request:Request):
    data = {
        "page": "Home page"
    }
    return templates.TemplateResponse(
        "index.html", {"request": request, "data": data}
    )

@app.post('/recommended_content/', response_class=HTMLResponse)
def recommended_content(request:Request, title: str = Form(...)):
    recommended_movies = content_recommender(title, cosine_sim=cosine_sim, df=movies_df, indices=indices_cont, top_n=10)
    return templates.TemplateResponse("index.html", {"request": request, "recommended_movies": recommended_movies})

@app.post('/recommended_collaborative/', response_class=HTMLResponse)
def recommended_content(request:Request, user_id:int = Form(...)):
    recommended_movies = collaborative_recommend(user_id, movies_df)
    return templates.TemplateResponse("index.html", {"request": request, "recommended_movies": recommended_movies})

@app.post('/recommended_hybrid/', response_class=HTMLResponse)
def recommended_content(request:Request, title: str = Form(...), user_id:int = Form(...)):
    recommended_movies = hybrid_recommend(user_id, title, svd=svd, cosine_sim=cosine_sim, movies_df=movies_df, indices_cont=indices_cont, top_n = 10)
    return templates.TemplateResponse("index.html", {"request": request, "recommended_movies": recommended_movies})

# if(__name__) == '__main__':
#         uvicorn.run(
#         "app:app",
#         host    = "0.0.0.0",
#         port    = 8036, 
#         reload  = True
#     )
