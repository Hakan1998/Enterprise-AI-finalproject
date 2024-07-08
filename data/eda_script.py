#Notwendige Libraries importieren
import pandas as pd #library for data manipulation and analysis#
import seaborn as sns #for data visualization#
import numpy as np #library for numerical computing#
import matplotlib.pyplot as plt #for data visualization#
from wordcloud import WordCloud #word cloud visualizations#

df = pd.read_csv("/workspaces/enterpriseai/data/movies_metadata.csv")
print("The shape of the dataset is : ", df.shape) #Um herauszufinden wieviel "rows" und "columns" der Datensatz umfasst
df.head(1) #Verständnis für verfügbare Spalten&Daten erlangen
df.info()
df.describe() # statistische Zusammenfassung der numerischen Spalten, um grundlegende Statistiken wie min/max, mean... zu erhalten
df.isnull().sum() #Untersuchung des Datensatzes auf fehlende Werte, da fehlende Werte die Robustheit und Zuverlässigkeit einschränken können#
df.budget.value_counts(dropna=False).head(20) #bedeutet bei 36573 Filme was das Budget 0/ ist das Budget nicht bekannt ist (NOCHMAL PRÜFEN)#
df['vote_average'].value_counts(dropna=False).head(20) # Bewertungen wie 0.0, 6.0, 5.0 und 7.0 kommen häufig vor, während andere Bewertungen weniger häufig sind --> Dies könnte darauf hinweisen, dass bestimmte Bewertungen tendenziell häufiger vergeben werden als andere#
df.describe(include='object') #Generierung eines besseres Überblicks#
df.hist(figsize= (18,10), bins =100)
plt.show()

#Visualisierung der Daten zur Erkennung von Mustern, Trends und Anomalien#

#Zusammenhang zwischen Budget des Films und des Umsatzes? Hypthese "Filme mit höherem Budget generieren höheren Umsatz"#
plt.figure(figsize=(10, 6))
plt.scatter(df['budget'], df['revenue'], alpha=0.5)
plt.title('Budget vs. Umsatz')
plt.xlabel('Budget (in Millionen $)')
plt.ylabel('Umsatz (in Millionen $)')
plt.show()

# Die 10 beliebtesten Filme im Dataset (nach Häufigkeit)
top_filme = df['title'].value_counts().head(10)
# Visualisierung der beliebtesten Filme
plt.figure(figsize=(10, 6))
top_filme.plot(kind='bar', color='skyblue')
plt.title('Die 10 beliebtesten Filme')
plt.xlabel('Filmtitel')
plt.ylabel('Anzahl')
plt.xticks(rotation=45, ha='right')
plt.show()
          
df[df.title == 'Cinderella'] # In dem Dataset befinden sich 11 Einträge zum Film "Cinderella"#

# Die 10 beliebtesten Genres (nach Häufigkeit)
top_genres = df['genres'].explode().value_counts().head(10)
# Visualisierung der beliebtesten Genres
plt.figure(figsize=(10, 6))
top_genres.plot(kind='bar', color='lightgreen')
plt.title('Die 10 beliebtesten Genres')
plt.xlabel('Genre')
plt.ylabel('Anzahl')
plt.xticks(rotation=45, ha='right')
plt.show()

# Die 10 häufigsten Sprachen
top_sprachen = df['original_language'].value_counts().head(10)
# Die 10 häufigsten Produktionsländer
top_laender = df['production_countries'].explode('production_countries').value_counts().head(10)

# Visualisierung der häufigsten Sprachen
plt.figure(figsize=(10, 6))
top_sprachen.plot(kind='bar', color='skyblue')
plt.title('Die 10 häufigsten Sprachen')
plt.xlabel('Sprache')
plt.ylabel('Anzahl')
plt.xticks(rotation=45, ha='right')
plt.show()

# Visualisierung der häufigsten Produktionsländer
plt.figure(figsize=(10, 6))
top_laender.plot(kind='bar', color='lightgreen')
plt.title('Die 10 häufigsten Produktionsländer')
plt.xlabel('Land')
plt.ylabel('Anzahl')
plt.xticks(rotation=45, ha='right')
plt.show()

# Histogramm der Filmlängen
plt.figure(figsize=(10, 6))
plt.hist(df['runtime'], bins=30, range=(0, 200), color='skyblue', edgecolor='black')
plt.title('Verteilung der Filmlängen')
plt.xlabel('Filmlänge (in Minuten)')
plt.ylabel('Anzahl der Filme')
plt.show()


df = pd.read_csv("/workspaces/enterpriseai/data/ratings_small.csv")
# Grundlegende Statistiken
print(df.describe())
print(df.info())

print(df.isnull().sum()) # Fehlende Werte --> keine fehlenden Werte

#Visualisierung der Daten zur Erkennung von Mustern, Trends und Anomalien#

# Verteilung der Bewertungen
plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], bins=10, kde=False)
plt.title('Verteilung der Bewertungen')
plt.xlabel('Bewertung')
plt.ylabel('Anzahl')
plt.show()

#Analyse des Nutzerverhaltens#
# Bewertungen pro Nutzer
ratings_per_user = df.groupby('userId').size().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
ratings_per_user.head(10).plot(kind='bar', color='lightgreen')
plt.title('Top 10 Nutzer nach Anzahl der Bewertungen')
plt.xlabel('Nutzer ID')
plt.ylabel('Anzahl der Bewertungen')
plt.show()

#zeitbasierte Analyse#

# Bewertungen über die Zeit
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('timestamp', inplace=True)
ratings_over_time = df.resample('M').size()
plt.figure(figsize=(10, 6))
ratings_over_time.plot()
plt.title('Anzahl der Bewertungen im Laufe der Zeit')
plt.xlabel('Zeit')
plt.ylabel('Anzahl der Bewertungen')
plt.show()

# Durchschnittliche Bewertung über die Zeit
average_rating_over_time = df.resample('M')['rating'].mean()
plt.figure(figsize=(10, 6))
average_rating_over_time.plot()
plt.title('Durchschnittliche Bewertung im Laufe der Zeit')
plt.xlabel('Zeit')
plt.ylabel('Durchschnittliche Bewertung')
plt.show()

#Filmbeliebtheit#

# Meist bewertete Filme
most_rated_movies = df.groupby('movieId').size().sort_values(ascending=False)[:10]

plt.figure(figsize=(10, 6))
most_rated_movies.plot(kind='bar', color='purple')
plt.title('Top 10 Meist bewertete Filme')
plt.xlabel('Film ID')
plt.ylabel('Anzahl der Bewertungen')
plt.show()
# Höchst bewertete Filme nach durchschnittlicher Bewertung
average_rating_per_movie = df.groupby('movieId')['rating'].mean()
highest_rated_movies = average_rating_per_movie.sort_values(ascending=False)[:10]

plt.figure(figsize=(10, 6))
highest_rated_movies.plot(kind='bar', color='orange')
plt.title('Top 10 Höchst bewertete Filme nach Durchschnittsbewertung')
plt.xlabel('Film ID')
plt.ylabel('Durchschnittliche Bewertung')
plt.show()

# Laden des Datensatzes
df_ratings = pd.read_csv("/workspaces/enterpriseai/data/ratings_small.csv")

# Anzahl der Bewertungen pro Film zählen
rating_counts = df_ratings['movieId'].value_counts().reset_index()
rating_counts.columns = ['movieId', 'rating_count']

# Visualisierung der Bewertungsverteilung
plt.figure(figsize=(12, 6))
sns.histplot(data=rating_counts, x='rating_count', bins=50, kde=True)
plt.title('Verteilung der Anzahl der Bewertungen pro Film')
plt.xlabel('Anzahl der Bewertungen')
plt.ylabel('Anzahl der Filme')
plt.show()

# Konvertierung des Zeitstempels in ein Datumsformat
df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s')

# Extrahieren des Jahres aus dem Zeitstempel
df_ratings['year'] = df_ratings['timestamp'].dt.year

# Berechnung der durchschnittlichen Bewertung pro Jahr
avg_rating_per_year = df_ratings.groupby('year')['rating'].mean().reset_index()

# Visualisierung der durchschnittlichen Bewertung über die Zeit
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='rating', data=avg_rating_per_year)
plt.title('Durchschnittliche Bewertung über die Zeit')
plt.xlabel('Jahr')
plt.ylabel('Durchschnittliche Bewertung')
plt.xticks(rotation=45)
plt.show()
