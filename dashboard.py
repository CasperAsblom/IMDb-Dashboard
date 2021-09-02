''' IMDb Data Analysis Dashboard
    Created on 2 sep. 2021
    @author: Casper Asblom'''


############### IMPORTING PACKAGES

# Data processing

import pandas as pd
import numpy as np
from collections import Counter

# Layout
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import dash_daq as daq

############### INITIALIZATION

app = dash.Dash(__name__, title="IMDb Dashboard", external_stylesheets=[dbc.themes.FLATLY])

pd.options.mode.chained_assignment = None # suppress SettingWithCopyWarning

############### IMPORTING DATA

ratings = pd.read_excel(r'new_ratings.xlsx', engine='openpyxl')
ratings.drop(ratings.columns[ratings.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # drop unnamed columns

# Merge data with IMDb extensive dataset (Available at: https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset?select=IMDb+movies.csv)

extended_ratings = pd.read_excel(r'IMDb-movies.xlsx', engine='openpyxl')
new_ratings = pd.merge(ratings, extended_ratings,  how='left', left_on=['Title','Year'], right_on = ['original_title','year'])
new_ratings.drop(['metascore', 'language'], axis=1, inplace=True)
new_ratings.dropna(inplace=True) # drop rows with at least 1 NaN

############### STYLE

# Colors

colors = {
    'background':'#161a28',
    'text': 'white',
    'purple': '#9B51E0',
    'yellow': '#f4d44d',
    'dark-blue':'#1e2130'
}

# Colorscales

PuYl = ["#9B51E0","#AD6BC3","#BF85A5","#D0A088","#F4D44D"]
PuYl23 = ["#9B51E0","#9F57D9","#A35DD3","#A763CC","#AB69C5","#AF6FBF","#B375B8","#B77BB1","#BB81AB","#BF87A4","#C38D9D","#C89397","#CC9890","#D09E89","#D4A482","#D8AA7C","#DCB075","#E0B66E","#E4BC68","#E8C261","#ECC85A","#F0CE54","#F4D44D"]
PuYl12 = ["#9B51E0","#A35DD3","#AB69C5","#B375B8","#BB81AB","#C38D9D","#CC9890","#D4A482","#DCB075","#E4BC68","#ECC85A","#F4D44D"]

############### DATA FOR GRAPHS

# Decade data

def generate_decade_data():
    year_rating_df = ratings[["Year", "Rating"]]

    bins = [1870, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
    labels = ['1800s', '1900s', '1910s', '1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']

    year_rating_df["Year"] = pd.cut(year_rating_df["Year"], bins, labels=labels)

    year_rating_df.columns = ["Decade", "Rating"] #rename to decade

    decades = year_rating_df["Decade"]
    decade_counts = Counter(decades)
    return year_rating_df, decade_counts

# Genre data

def generate_genre_data():
    genres = ratings["Genres"]
    split_genres = genres.str.split(', ')
    all_genres = [item for sublist in split_genres for item in sublist]
    genre_counts = Counter(all_genres)

    return genre_counts

############### GRAPHS

# Map graph

def generate_map_graph():
    countries = new_ratings["country"]
    split_countries = countries.str.split(', ')
    all_countries = [item for sublist in split_countries for item in sublist]
    country_counts = Counter(all_countries)

    countries = []
    n_movies = []

    for x,y in country_counts.items():
            countries.append(x)
            n_movies.append(y)

    avg_ratings = []

    for country in countries:
        country_ratings = new_ratings.loc[new_ratings.country.str.contains(country), 'Rating'].values
        avg_rating = sum(country_ratings)/len(country_ratings)
        avg_ratings.append(round(avg_rating,2))
        
    country_movies = pd.DataFrame({"Country":countries,
                                    "Watched movies":n_movies,
                                    "Average rating":avg_ratings})


    gapminder = country_movies
    gapminder['counts'] = np.random.uniform(low=1, high=sum(country_movies["Watched movies"]), size=len(gapminder)).tolist()

    map_graph = px.choropleth(gapminder, locations="Country",
                        locationmode='country names',
                        color="Watched movies", 
                        hover_name="Country",
                        hover_data=country_movies[["Country","Watched movies", "Average rating"]],
                        color_continuous_scale=PuYl[::-1])

    map_graph.update_layout(
        margin=dict(t=0),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    return map_graph

# Ratings histogram

def generate_ratings_histogram():
    ratings_histogram = px.histogram(ratings["Rating"], x="Rating")
    ratings_histogram.update_layout(
        yaxis_title="Movies",
        font=dict(
            size=13
        ),
        margin=dict(t=0, b=0, l=0),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    ratings_histogram.update_traces(marker_color=colors['purple'], marker_line_color=colors['background'],
                    marker_line_width=1.5)
    return ratings_histogram

# Decades ordered by average rating

def generate_decades_avg_rating_graph():
    year_rating_df, decade_counts = generate_decade_data()

    decades_df = []
    for x,y in decade_counts.items():
        if y>=10: # only includes decades with at least 10 watched movies
            decades_df.append(x)
        
        
    avg_decade_ratings = []
    avg_decade_ratings_std = []

    for decade in decades_df:
        decade_ratings = year_rating_df.loc[year_rating_df.Decade.str.contains(decade), 'Rating'].values
        avg_decade_rating = sum(decade_ratings)/len(decade_ratings)
        avg_decade_ratings.append(avg_decade_rating)
        avg_decade_ratings_std.append(np.std(decade_ratings))
        

    avg_decade_ratings_df = pd.DataFrame({"Decade":decades_df,
                                        "Average rating":avg_decade_ratings,
                                        "Standard deviation":avg_decade_ratings_std})


    sorted_avg_decade_ratings = avg_decade_ratings_df.sort_values(by="Average rating", ascending=False)

    decades_average_rating_graph = go.Figure(data=go.Scatter(
            x=sorted_avg_decade_ratings["Decade"],
            y=sorted_avg_decade_ratings["Average rating"],
            mode="markers",
            #line=dict(color="black"),
            error_y=dict(
                type='data',
                symmetric=True,
                color=colors['purple'],
                array=sorted_avg_decade_ratings["Standard deviation"])
            ))
    decades_average_rating_graph.update_layout(
        yaxis_title="Average rating",
        margin=dict(t=10, l=0),
        font=dict(
            size=13,
            color="black"
        ),
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    decades_average_rating_graph.update_traces(marker_color='#444444',
                    marker_size=15)

    return decades_average_rating_graph

# Decade distribution piechart

def generate_decade_piechart():
    year_rating_df, decade_counts = generate_decade_data()

    decades_df = []
    watched_movies = []

    for x,y in decade_counts.items():
        decades_df.append(x)
        watched_movies.append(y)

    decade_piechart_df = pd.DataFrame({"Decade":decades_df,
                            "Watched movies":watched_movies})

    decade_piechart_df_sorted = decade_piechart_df.sort_values("Watched movies")

    decade_piechart_graph = px.pie(decade_piechart_df_sorted, values='Watched movies', names='Decade')
    decade_piechart_graph.update_traces(textposition='inside', textinfo="percent+label", textfont_size=20,  marker=dict(colors=PuYl12, line=dict(color=colors['background'], width=0.5)))
    decade_piechart_graph.update_layout(
        font=dict(size=20, color='black'),
        margin=dict(t=10),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    return decade_piechart_graph

# Genres ordered by average rating

def generate_genres_avg_rating_graph():
    genre_counts = generate_genre_data()

    unique_genres = []

    for x,y in genre_counts.items():
        if y>10: # only includes genres with at least 10 watched movies
            unique_genres.append(x)
        
        
    avg_genre_ratings = []
    avg_genre_ratings_std = []

    for genre in unique_genres:
        genre_ratings = ratings.loc[ratings.Genres.str.contains(genre), 'Rating'].values
        avg_genre_rating = sum(genre_ratings)/len(genre_ratings)
        avg_genre_ratings.append(avg_genre_rating)
        avg_genre_ratings_std.append(np.std(genre_ratings))

    avg_genre_ratings_df = pd.DataFrame({"Genre":unique_genres,
                                        "Average rating":avg_genre_ratings,
                                        "Standard deviation":avg_genre_ratings_std})


    sorted_avg_genre_ratings = avg_genre_ratings_df.sort_values(by="Average rating", ascending=False)

    genre_average_rating_graph = go.Figure(data=go.Scatter(
            x=sorted_avg_genre_ratings["Genre"],
            y=sorted_avg_genre_ratings["Average rating"],
            mode='markers',
            error_y=dict(
                type='data',
                symmetric=True,
                color=colors['purple'],
                array=sorted_avg_genre_ratings["Standard deviation"])
            ))
    genre_average_rating_graph.update_layout(
        margin=dict(t=10, l=0),
        yaxis_title="Average rating",
        font=dict(
            size=13,
            color="black"
        ),
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    genre_average_rating_graph.update_traces(marker_color='#444444',
                    marker_size=15)
                    
    return genre_average_rating_graph

# Genre distribution pie chart

def generate_genre_piechart():
    genre_counts = generate_genre_data()

    genres = []
    watched_movies = []
    for x,y in genre_counts.items():
        genres.append(x)
        watched_movies.append(y)

    plotly_piechart_df = pd.DataFrame({"Genre":genres,
                            "Watched movies":watched_movies})

    plotly_piechart_df_sorted = plotly_piechart_df.sort_values("Watched movies")

    genre_piechart = px.pie(plotly_piechart_df_sorted, values='Watched movies', names='Genre')
    genre_piechart.update_traces(textposition='inside', textinfo="percent+label", textfont_size=20, marker=dict(colors=PuYl23, line=dict(color=colors['background'], width=0.5)))
    genre_piechart.update_layout(
        margin=dict(t=10),
        font=dict(size=20),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    return genre_piechart

# Favorite directors based on average rating

def generate_favorite_directors():
    directors = ratings["Directors"]
    split_directors = directors.str.split(', ')
    all_directors = [item for sublist in split_directors for item in sublist]
    counts = Counter(all_directors)

    top_directors = []
    avg_ratings = []

    for x,y in counts.items():
        if y >= 5: # TODO: find a good metric here... some great directors are excluded due to >=5 movies exclusion
            top_directors.append(x)

    for director in top_directors:
        director_ratings = ratings.loc[ratings.Directors.str.contains(director), 'Rating'].values
        avg_rating = sum(director_ratings)/len(director_ratings)
        avg_ratings.append(avg_rating)

    avg_director_ratings = pd.DataFrame({"Director":top_directors,
                                        "Average rating":avg_ratings})

    sorted_avg_director_ratings = avg_director_ratings.sort_values(by="Average rating", ascending=False)

    favorite_directors_graph = go.Figure(go.Bar(
                x=sorted_avg_director_ratings["Average rating"][:10],
                y=sorted_avg_director_ratings["Director"][:10],
                orientation='h'))

    favorite_directors_graph.update_layout(
        xaxis_title="Average rating",
        margin=dict(t=30, r=0),
        font=dict(size=13, color='black'),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    favorite_directors_graph.update_traces(marker_color=colors['purple'],
                    marker_line_width=1.5, marker_line_color=colors['background'])

    favorite_directors_graph.update_xaxes(range=[min(sorted_avg_director_ratings["Average rating"][:10])-0.5, max(sorted_avg_director_ratings["Average rating"][:10])+0.5])
        
    favorite_directors_graph['layout']['yaxis']['autorange'] = "reversed"

    return favorite_directors_graph

# Favorite actors based on average rating

def generate_favorite_actors():
    actors = new_ratings["actors"]
    split_actors = actors.str.split(', ')
    all_actors = [item for sublist in split_actors for item in sublist]
    counts = Counter(all_actors)

    top_actors = []
    avg_ratings = []

    for x,y in counts.items():
        if y >= 10: # TODO: find a good metric here... some great actors are excluded due to >=10 movies exclusion
            top_actors.append(x)

    for actor in top_actors:
        actor_ratings = new_ratings.loc[new_ratings.actors.str.contains(actor), 'Rating'].values
        avg_rating = sum(actor_ratings)/len(actor_ratings)
        avg_ratings.append(avg_rating)

    avg_actor_ratings = pd.DataFrame({"Actor":top_actors,
                                        "Average rating":avg_ratings})

    sorted_avg_actor_ratings = avg_actor_ratings.sort_values(by="Average rating", ascending=False)

    favorite_actors_graph = go.Figure(go.Bar(
                x=sorted_avg_actor_ratings["Average rating"][:10],
                y=sorted_avg_actor_ratings["Actor"][:10],
                orientation='h'))

    favorite_actors_graph.update_layout(
        xaxis_title="Average rating",
        margin=dict(t=30, r=0),
        font=dict(size=13, color='black'),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    favorite_actors_graph.update_traces(marker_color=colors['yellow'],
                    marker_line_width=1.5, marker_line_color=colors['background'])

    favorite_actors_graph.update_xaxes(range=[min(sorted_avg_actor_ratings["Average rating"][:10])-0.5, max(sorted_avg_actor_ratings["Average rating"][:10])+0.5])

    favorite_actors_graph['layout']['yaxis']['autorange'] = "reversed"

    return favorite_actors_graph

# Favorite writers based on average rating

def generate_favorite_writers():
    writers = new_ratings["writer"]
    split_writers = writers.str.split(', ')
    all_writers = [item for sublist in split_writers for item in sublist]
    counts = Counter(all_writers)

    top_writers = []
    avg_ratings = []

    for x,y in counts.items():
        if y >= 5: # TODO: find a good metric here... some great writers are excluded due to >=5 movies exclusion
            top_writers.append(x)

    for writer in top_writers:
        writer_ratings = new_ratings.loc[new_ratings.writer.str.contains(writer), 'Rating'].values
        avg_rating = sum(writer_ratings)/len(writer_ratings)
        avg_ratings.append(avg_rating)

    avg_writer_ratings = pd.DataFrame({"Writer":top_writers,
                                        "Average rating":avg_ratings})

    sorted_avg_writer_ratings = avg_writer_ratings.sort_values(by="Average rating", ascending=False)

    favorite_writers_graph = go.Figure(go.Bar(
                x=sorted_avg_writer_ratings["Average rating"][:10],
                y=sorted_avg_writer_ratings["Writer"][:10],
                orientation='h'))

    favorite_writers_graph.update_layout(
        xaxis_title="Average rating",
        margin=dict(t=30,r=0),
        font=dict(size=13, color='black'),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    favorite_writers_graph.update_traces(marker_color=colors['purple'],
                    marker_line_width=1.5, marker_line_color=colors['background'])

    favorite_writers_graph.update_xaxes(range=[min(sorted_avg_writer_ratings["Average rating"][:10])-0.5, max(sorted_avg_writer_ratings["Average rating"][:10])+0.5])

    favorite_writers_graph['layout']['yaxis']['autorange'] = "reversed"

    return favorite_writers_graph

# Favorite production companies based on average rating

def generate_favorite_pcomps():
    pcomps = new_ratings["production_company"]
    counts = Counter(pcomps)

    top_pcomps = []
    avg_ratings = []

    for x,y in counts.items():
        if y >= 15: # TODO: find a good metric here... some great pcomps are excluded due to >=15 movies exclusion
            top_pcomps.append(x)
            

    for pcomp in top_pcomps:
        pcomp_ratings = new_ratings.loc[new_ratings.production_company.str.contains(pcomp, regex=False), 'Rating'].values # regex=False is to avoid error when pcomp contains paranthesis, as with e.g. Metro-Goldwyn-Mayer (MGM)
        avg_rating = sum(pcomp_ratings)/len(pcomp_ratings)
        avg_ratings.append(avg_rating)
        
        
    avg_pcomp_ratings = pd.DataFrame({"Production company":top_pcomps,
                                        "Average rating":avg_ratings})

    sorted_avg_pcomp_ratings = avg_pcomp_ratings.sort_values(by="Average rating", ascending=False)

    favorite_pcomps_graph = go.Figure(go.Bar(
                x=sorted_avg_pcomp_ratings["Average rating"][:10],
                y=sorted_avg_pcomp_ratings["Production company"][:10],
                orientation='h'))

    favorite_pcomps_graph.update_layout(
        xaxis_title="Average rating",
        margin=dict(t=30,r=0),
        font=dict(size=13, color='black'),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    favorite_pcomps_graph.update_traces(marker_color=colors['yellow'],
                    marker_line_width=1.5, marker_line_color=colors['background'])
        
    favorite_pcomps_graph.update_xaxes(range=[min(sorted_avg_pcomp_ratings["Average rating"][:10])-0.5, max(sorted_avg_pcomp_ratings["Average rating"][:10])+0.5])

    favorite_pcomps_graph['layout']['yaxis']['autorange'] = "reversed"

    return favorite_pcomps_graph

############### LAYOUT


app.layout = dbc.Container(style={'background-image':'url(https://images.squarespace-cdn.com/content/v1/5a7b6485be42d67e9c852f09/1518285038853-DH7MTETF6IG40CIDXGSQ/data.jpg?format=2500w)', 'height':'100vh', 'width':'100vw'}, children = [dbc.Container(style={'backgroundColor': colors['background'], 'color': colors['text'], 'fontFamily': "Open Sans; sans-serif", 'opacity':0.95, 'height':'100vh', 'width':'100vw', 'margin-left':-15, 'margin-right':15}, children=
    [
        dbc.Row(dbc.Col(html.H1('IMDb Data Analysis Dashboard', className='text-center text-primary, mb-3'), style={'backgroundColor': colors['dark-blue'], 'padding-top':20, 'padding-bottom':20})),  # header row

        dbc.Row([  # start of second row
            dbc.Col([  # first column on second row
            html.H4('Country Distribution', className='text-center', style={'margin-top':25}),
            dcc.Graph(id='map-graph',
                      figure=generate_map_graph(),
                      style={'height':550},
                      config= {'displaylogo': False})
            ], width={'size': 8, 'offset': 0, 'order': 1}),  # width first column on second row
            dbc.Col([  # second column on second row
            html.Div([
                html.H4('Watched Movies', className='text-center'),
                daq.LEDDisplay(
                id="movies-led",
                value=len(ratings["Title"]),
                color=colors['yellow'],
                backgroundColor='rgba(30, 33, 48, 0.5)',
                size=50)
            ], style={'width': '49%', 'display': 'inline-block', 'textAlign':'center', 'margin-bottom':20, 'margin-top':25}),
            html.Div([
                html.H4('Average Rating', className='text-center'),
                daq.LEDDisplay(
                id="rating-led",
                value=f"{round(np.mean(ratings['Rating']), 2)}",
                color=colors['yellow'],
                backgroundColor='rgba(30, 33, 48, 0.5)',
                size=50)
            ], style={'width': '49%', 'display': 'inline-block', 'textAlign':'center', 'margin-bottom':20, 'margin-top':25}),
            html.H4('Rating Distribution', className='text-center'),
            dcc.Graph(id='ratings-histogram',
            figure=generate_ratings_histogram(),
            style={'height':350},
            config= {'displaylogo': False})
            ], width={'size': 4, 'offset': 0, 'order': 2}),  # width second column on second row

        ]),  # end of second row

        dbc.Row([  # start of third row
            dbc.Col([  # first column on third row
                html.H4('Decades ordered by average rating', className='text-center', style={'position':'relative'}),

                html.Div([
                    html.Span(
                    "?",
                    id="tooltip-target",
                    style={
                            "textAlign": "center", 
                            "color": "white",
                            "height":25,
                            "width":25,
                            "background-color":"#bbb",
                            "border-radius":"50%",
                            "display":"inline-block",
                            "position":"absolute",
                            "top":0,
                            "left":"70%"
                    }, className="dot"),

                    dbc.Tooltip(
                        "Only includes decades with at least 10 watched movies",
                        target="tooltip-target",
                )]),                    

                dcc.Graph(id='decades-average-rating-graph',
                      figure = generate_decades_avg_rating_graph(),
                      style={'height':380},
                      config= {'displaylogo': False}),
            ], width={'size': 8, 'offset': 0, 'order': 1}),  # width first column on third row
            dbc.Col([  # second column on third row
                html.H4('Decade Distribution', className='text-center'),
                dcc.Graph(id='decade-piechart-graph',
                      figure = generate_decade_piechart(),
                      style={'height':380, 'margin-left':-115},
                      config= {'displaylogo': False}),
            ], width={'size': 4, 'offset': 0, 'order': 2}),  # width second column on third row
        ]),  # end of third row
        
        dbc.Row([  # start of fourth row
            dbc.Col([  # first column on fourth row
                html.H4('Genres ordered by average rating', className='text-center', style={'position':'relative'}),
                html.Div([
                    html.Span(
                    "?",
                    id="tooltip-target2",
                    style={
                            "textAlign": "center", 
                            "color": "white",
                            "height":25,
                            "width":25,
                            "background-color":"#bbb",
                            "border-radius":"50%",
                            "display":"inline-block",
                            "position":"absolute",
                            "top":0,
                            "left":"70%"
                    }, className="dot"),

                    dbc.Tooltip(
                        "Only includes genres with at least 10 watched movies",
                        target="tooltip-target2",
                )]),                    

                dcc.Graph(id='genre-average-rating-graph',
                      figure=generate_genres_avg_rating_graph(),
                      style={'height':380},
                      config= {'displaylogo': False}),
            ], width={'size': 8, 'offset': 0, 'order': 1}),  # width first column on fourth row
            dbc.Col([  # second column on fourth row
                html.H4('Genre Distribution', className='text-center'),
                dcc.Graph(id='genre-piechart',
                      figure = generate_genre_piechart(),
                      style={'height':380,  'margin-left':-50},
                      config= {'displaylogo': False}),
            ], width={'size': 4, 'offset': 0, 'order': 2}),  # width second column on fourth row
        ]),  # end of fourth row

        dbc.Row([  # start of fifth row
            dbc.Col([  # first column on fifth row
                html.H4('Favorite directors', className='text-center'),
                dcc.Graph(id='favorite-directors-graph',
                      figure=generate_favorite_directors(),
                      style={'height':380},
                      config= {'displaylogo': False}),
            ], width={'size': 3, 'offset': 0, 'order': 1}),  # width first column on fifth row
            dbc.Col([  # second column on fifth row
                html.H4('Favorite actors', className='text-center'),
                dcc.Graph(id='favorite-actors-graph',
                      figure = generate_favorite_actors(),
                      style={'height':380},
                      config= {'displaylogo': False}),
            ], width={'size': 3, 'offset': 0, 'order': 2}),  # width second column on fifth row
            dbc.Col([  # third column on fifth row
                html.H4('Favorite writers', className='text-center'),
                dcc.Graph(id='favorite-writers-graph',
                      figure = generate_favorite_writers(),
                      style={'height':380},
                      config= {'displaylogo': False}),
            ], width={'size': 3, 'offset': 0, 'order': 3}),  # width third column on fifth row          
            dbc.Col([  # fourth column on fifth row
                html.H4('Favorite production companies', className='text-center'),
                dcc.Graph(id='favorite-pcomps-graph',
                      figure = generate_favorite_pcomps(),
                      style={'height':380},
                      config= {'displaylogo': False}),
            ], width={'size': 3, 'offset': 0, 'order': 4}),  # width fourth column on fifth row        
        ]) # end of fifth row    

    ], fluid=True)], fluid=True)


if __name__ == "__main__":
    app.run_server(debug=False, port=8058)
