import streamlit as st
import pandas as pd
import numpy as np
import datetime
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import nltk
nltk.download('punkt')

from google_play_scraper import Sort, reviews_all

from backend import text_filtering, stopstem, word_tokenize_wrapper
from backend import pembakuan_kata, stop, hitung_kata, ngrams
from backend import convertdf_tocsv


@st.cache_data
def scrape_reviews():
  tiket_review = reviews_all(
      'com.tiket.gits',
      sleep_milliseconds=0,
      lang='id',
      country='id',
      sort=Sort.NEWEST
  )
  df = pd.DataFrame(tiket_review)
  df['date'] = pd.to_datetime(df['at'], format = '%Y%m%d') 
  return df

def date_selector(start_date, end_date,df):
  df_filter = df.loc[((df['date'] >= start_date) & (df['date'] <= end_date))]
  return df_filter

svm_model = joblib.load('app/svm_ready.joblib')
vectorizer = joblib.load('app/vectorizer_svm_ready.joblib')


def main():
  activities = ["Main Page", "Tentang Kami"]
  choice = st.sidebar.selectbox("Choice", activities)

  if choice == 'Main Page':
    st.title("Aplikasi Analisis Sentimen")
    st.markdown("Aplikasi ini dibuat untuk analisis sentimen")

    st.subheader("Scrap Data Ulasan dari Google Play Store")

    col1, col2 = st.columns (2)
    with col1:
      d = st.date_input("Tanggal awal", datetime.date(2023,1,1))
    with col2:
      f = st.date_input("Tanggal akhir", datetime.date(2023,2,1))
    d = d.strftime('%Y-%m-%d')
    f = f.strftime('%Y-%m-%d')

    if st.button("Mulai scraping"):
      df = scrape_reviews()
      df_filter = date_selector(d,f,df)
      
      st.markdown('**Data hasil scraping**')
      st.dataframe(df_filter)

      fig1 = px.histogram(df_filter, x='score', title='Sebaran rating')
      fig1.update_layout(showlegend=False)
      fig1.update_layout(bargap=0.1)
      fig1.update_traces(marker=dict(color='#50C878'))
      st.plotly_chart(fig1)

      dfprep = df_filter[['content','score']]
      dfprep['filtered'] = dfprep['content'].apply(text_filtering)
      dfprep['cleaned'] = dfprep['filtered'].apply(stopstem)
      dfprep['tokens'] = dfprep['cleaned'].apply(word_tokenize_wrapper)
      
      kamusslang = pd.read_csv("app/kamus_slangwords.csv")
      kata_pembakuan_dict = {}
      for index, row in kamusslang.iterrows():
        if row[0] not in kata_pembakuan_dict:
          kata_pembakuan_dict[row[0]] = row[1]
      
      dfprep['tokens_pembakuan'] = dfprep['tokens'].apply(pembakuan_kata)
      dfprep['pembakuan'] = dfprep['tokens_pembakuan'].apply(lambda x:' '.join(x))

      dfprep['pembakuan_bersih'] = dfprep['pembakuan'].apply(stop)
      dfprep.dropna(inplace=True)
      dfprep['temp_token'] = dfprep['pembakuan_bersih'].apply(lambda x: word_tokenize(str(x)))

      st.markdown("**Hasil preprocessing data**")
      st.dataframe(dfprep)

      st.markdown("**Hasil analisis sentimen**")
      text_vector = vectorizer.transform(dfprep['pembakuan_bersih'].apply(lambda x: str(x)))
      dfprep['predicted_sentiment'] = svm_model.predict(text_vector)
      dfresult = dfprep[['content', 'pembakuan_bersih','predicted_sentiment','temp_token']]
      st.dataframe(dfresult[['content', 'pembakuan_bersih','predicted_sentiment']])

      dfresultpos = dfresult[dfresult['predicted_sentiment'] == 1]
      dfresultneg = dfresult[dfresult['predicted_sentiment'] == 0]

      #count words
      dffreq = hitung_kata(dfresult)
      dffreqs = dffreq.reset_index()

      #count words positive
      dffreqpos = hitung_kata(dfresultpos)
      dffreqposres = dffreqpos.reset_index()
      dffreqpostop = dffreqposres.head(10)
      
      #count words negative
      dffreqneg = hitung_kata(dfresultneg)
      dffreqnegres = dffreqneg.reset_index()
      dffreqnegtop = dffreqnegres.head(10)
    

      #plot count words positive
      fig2pos = go.Figure()
      fig2pos.add_trace(go.Bar(
              x=dffreqpostop['freq'],
              y=dffreqpostop['token'],
              orientation='h',
              marker=dict(color='#007AFF')
            ))
      fig2pos.update_layout(
              title='Kata yang paling sering muncul pada ulasan positif',
              xaxis_title='Frekuensi',
              yaxis_title='Kata',
              bargap=0.1
            )
      fig2pos.update_yaxes(autorange="reversed")
      st.plotly_chart(fig2pos)

      #plot count words negative
      fig2neg = go.Figure()
      fig2neg.add_trace(go.Bar(
              x=dffreqnegtop['freq'],
              y=dffreqnegtop['token'],
              orientation='h',
              marker=dict(color='#EE4B2B')
            ))
      fig2neg.update_layout(
              title='Kata yang paling sering muncul pada ulasan negatif',
              xaxis_title='Frekuensi',
              yaxis_title='Kata',
              bargap=0.1
            )
      fig2neg.update_yaxes(autorange="reversed")
      st.plotly_chart(fig2neg)

      #Wordcloud
      reviews_text = ' '.join(dfprep['pembakuan_bersih'].tolist())
      wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews_text)

      plt.figure(figsize=(10, 5))
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.title("Wordcloud")
      plt.axis('off')
      st.pyplot(plt)

      #Analisis Ngrams
      dfprep['bigrams'] = dfprep['temp_token'].apply(ngrams,n=2)
      df2 = hitung_kata(dfprep, 'bigrams').head()
      df3 = df2.reset_index()

      fig3 = px.bar(df3, x='token', y='freq', title='Analisis ngrams')
      fig3.update_xaxes(tickangle=-90)
      st.plotly_chart(fig3)

      #Analisis Ngrams positif
      dfresultpos['bigrams'] = dfresultpos['temp_token'].apply(ngrams, n=2)
      dfngrampos = hitung_kata(dfresultpos, 'bigrams').head()
      dfngrampos2 = dfngrampos.reset_index()

      fig4 = px.bar(dfngrampos2, x='token', y='freq', title = 'Analisis Ngrams Sentimen Positif')
      fig4.update_xaxes(tickangle =-90)
      fig4.update_traces(marker_color = '#007AFF')
      st.plotly_chart(fig4)

      #Analisis Ngrams negatif
      dfresultneg['bigrams'] = dfresultneg['temp_token'].apply(ngrams, n=2)
      dfngramneg = hitung_kata(dfresultneg, 'bigrams').head()
      dfngramneg2 = dfngramneg.reset_index()

      fig5 = px.bar(dfngramneg2, x='token', y='freq', title = 'Analisis Ngrams Sentimen Negatif')
      fig5.update_xaxes(tickangle=-90)
      fig5.update_traces(marker_color = '#EE4B2B')
      st.plotly_chart(fig5)

      positive_proportion = dfresult['predicted_sentiment'].mean() * 100

      st.subheader('Summary')
      col1, col2, col3 = st.columns(3)
      col1.metric("Rata-rata rating", round(df_filter.score.mean(),2))
      col2.metric("Kata Teratas", dffreqs['token'][0])
      col3.metric("Proporsi Sentimen Positif", f"{positive_proportion:.2f}%")

      st.markdown('**Download Data**')
      col4, col5 = st.columns(2)
      with col4:
        st.write("Download Data Hasil Scraping")
        st.download_button(label = "Download", data=convertdf_tocsv(df_filter), file_name = 'df_filter.csv', mime = 'text/csv')

      with col5:
        st.write("Download Data Analisis Sentimen")
        st.download_button(label = "Download", data=convertdf_tocsv(dfresult), file_name = 'dfresult.csv', mime = 'text/csv')
      



  elif choice == 'Tentang Kami':
    st.title("Tentang Kami")
    st.markdown("Terima kasih telah menggunakan aplikasi ini.")
    st.markdown("Anda menggunakan aplikasi sentimen analisis versi beta I")


if __name__ == '__main__':
  main()
