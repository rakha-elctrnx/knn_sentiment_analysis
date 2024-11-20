import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from core.crawling import YouTubeCrawler
from core.preprocessing import TextPreprocessor
from core.modeling import SentimentClassifier
from database.connection import DatabaseManager

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Inisialisasi komponen
    crawler = YouTubeCrawler()
    preprocessor = TextPreprocessor()
    classifier = SentimentClassifier()
    db_manager = DatabaseManager()

    # Sidebar
    st.sidebar.title("Sentiment Analysis App")
    menu = st.sidebar.radio("Menu", ["Crawling Content", "Data Training", "Data Test", "Hasil Analisis Sentimen"])

    # Menu: Crawling Content
    if menu == "Crawling Content":
        st.title("Crawling Content")
        
        # Masukkan ID Video yang diinginkan untuk crawling komentar
        video_id = st.text_input("Masukkan Video ID YouTube")
        
        if video_id:
            try:
                # Dapatkan detail video
                video_details = crawler.get_video_details(video_id)
                print(video_details)
                
                if video_details:
                    title, thumbnail_url = video_details
                    
                    # Simpan video ke database
                    insert_query = """
                        INSERT INTO youtube_video (video_id, title, thumbnail_url)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE title=%s, thumbnail_url=%s
                    """
                    params = (video_id, title, thumbnail_url, title, thumbnail_url)
                    print(params)

                    db_manager.execute_query(insert_query, params)

                    
                    # Ambil komentar
                    comments = crawler.get_comments(video_id)
                    
                    if comments:
                        # Simpan komentar ke database
                        insert_comments_query = """
                            INSERT INTO youtube_comments (video_id, comment, sender) 
                            VALUES (%s, %s, %s)
                        """
                        for comment, sender in comments:
                            params = (video_id, comment, sender)
                            db_manager.execute_query(insert_comments_query, params)

                        
                        st.success("Data video dan komentar berhasil disimpan ke database!")
                        
                        # Menampilkan data video dan komentar yang disimpan dalam bentuk tabel
                        st.subheader("Hasil Crawling Comments")
                        data_crawled = db_manager.fetch_data("""
                            SELECT * FROM youtube_comments 
                            WHERE video_id = %s
                        """, (video_id,))
                        st.dataframe(data_crawled)
                    else:
                        st.warning("Tidak ada komentar yang ditemukan.")
                else:
                    st.error("Tidak dapat menemukan informasi video.")
            
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                logger.error(f"Crawling error: {e}")

    # Menu: Data Training
    elif menu == "Data Training":
        st.title("Data Training")

        uploaded_file = st.file_uploader("Upload CSV untuk menambahkan Data Training", type="csv")
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                
                # Simpan data training ke database
                insert_query = "INSERT INTO training (comment, sentiment) VALUES (%s, %s)"
                for _, row in data.iterrows():
                    db_manager.execute_query(insert_query, (row['comment'], row['sentiment']))
                
                st.success("Data Training berhasil ditambahkan!")

            except Exception as e:
                st.error(f"Kesalahan saat mengunggah data: {e}")
                logger.error(f"Training data upload error: {e}")

        # Tampilkan data training
        data_train = db_manager.fetch_data("SELECT * FROM training")
        st.dataframe(data_train)

        st.title("Preprocessing Data Training")
        
        # Cek data preprocessed
        data_preprocessed = db_manager.fetch_data("SELECT * FROM preprocessed_training")
        
        if not data_preprocessed.empty:
            st.dataframe(data_preprocessed)
        else:
            if st.button("Mulai Preprocessing"):
                try:
                    # Ambil data training mentah
                    raw_data = db_manager.fetch_data("SELECT * FROM training")
                    
                    # Preprocessing
                    preprocessed_data = preprocessor.preprocess_texts(raw_data['comment'])
                    preprocessed_df = pd.DataFrame({
                        "text": preprocessed_data,
                        "sentiment": raw_data["sentiment"]
                    })
                    
                    # Simpan data preprocessed
                    insert_preprocessed_query = "INSERT INTO preprocessed_training (text, sentiment) VALUES (%s, %s)"
                    for _, row in preprocessed_df.iterrows():
                        db_manager.execute_query(insert_preprocessed_query, (row['text'], row['sentiment']))
                    
                    st.success("Preprocessing selesai!")
                    st.dataframe(preprocessed_df)
                
                except Exception as e:
                    st.error(f"Kesalahan preprocessing: {e}")
                    logger.error(f"Preprocessing error: {e}")

    # Menu: Data Test
    elif menu == "Data Test":
        st.title("Data Test")

        # Ambil video yang tersedia
        data_video = db_manager.fetch_data("SELECT * FROM youtube_video")
        video_ids = data_video["video_id"].unique()
        selected_video_id = st.selectbox("Pilih Video", video_ids)

        if selected_video_id:
            try:
                # Ambil komentar
                comments_data = db_manager.fetch_data("""
                    SELECT vc.comment_id, vc.sentiment, vc.comment, vc.sender, vc.video_id
                    FROM youtube_comments vc
                    JOIN youtube_video v ON vc.video_id = v.video_id
                    WHERE vc.video_id = %s
                """, (selected_video_id,))
                
                # Detail video
                selected_video_data = data_video[data_video["video_id"] == selected_video_id].iloc[0]
                title = selected_video_data["title"]
                thumbnail_url = selected_video_data["thumbnail_url"]

                st.subheader("Video Detail")
                st.write(f"**Title**: {title}")
                st.image(thumbnail_url, width=300)

                if not comments_data.empty:
                    st.subheader(f"Comments ({len(comments_data)})")
                    st.dataframe(comments_data.drop(columns=["video_id"]))

                    # Cek apakah kolom sentimen sudah terisi
                    if comments_data["sentiment"].isnull().any():
                        st.warning("Klasifikasi sentimen secara manual belum diisi.")
                        
                        # Opsi unggah CSV untuk update sentimen
                        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
                        
                        if uploaded_file:
                            try:
                                uploaded_data = pd.read_csv(uploaded_file)
                                
                                # Validasi kolom
                                if "sentiment" in uploaded_data.columns and "comment_id" in uploaded_data.columns:
                                    # Update sentimen manual
                                    update_query = """
                                        UPDATE youtube_comments 
                                        SET sentiment = %s 
                                        WHERE comment_id = %s
                                    """
                                    for _, row in uploaded_data.iterrows():
                                        db_manager.execute_query(update_query, (row['sentiment'], row['comment_id']))
                                    
                                    st.success("Sentimen berhasil diperbarui!")
                                
                                else:
                                    st.error("File CSV harus memiliki kolom 'comment_id' dan 'sentiment'.")
                            
                            except Exception as e:
                                st.error(f"Kesalahan mengunggah file: {e}")
                                logger.error(f"File upload error: {e}")

                    data_preprocessed = db_manager.fetch_data("""
                        SELECT text
                        FROM preprocessed_test
                        WHERE video_id = %s
                    """, (selected_video_id,))

                    if not data_preprocessed.empty:
                        st.subheader("Preprocessed Comments")
                        st.dataframe(data_preprocessed)
                    else:
                        if st.button("Mulai Preprocessing"):
                            preprocessed_data = preprocessor.preprocess_texts(comments_data["comment"])
                            preprocessed_df = pd.DataFrame({
                                "text": preprocessed_data,
                                "video_id": comments_data["video_id"]
                            })

                            insert_preprocessed_query = "INSERT INTO preprocessed_test (text, video_id) VALUES (%s, %s)"
                            for _, row in preprocessed_df.iterrows():
                                db_manager.execute_query(insert_preprocessed_query, (row['text'], row['video_id']))
                            
                            st.success("Preprocessing selesai!")
                            st.subheader("Preprocessed Comments")
                            st.dataframe(preprocessed_df)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                logger.error(f"Data test error: {e}")

    # Menu: Hasil Analisis Sentimen
    elif menu == "Hasil Analisis Sentimen":
        st.title("Hasil Analisis Sentimen")

        # Ambil video yang tersedia
        data_video = db_manager.fetch_data("SELECT * FROM youtube_video")
        video_ids = data_video["video_id"].unique()
        selected_video_id = st.selectbox("Pilih Video", video_ids)

        if selected_video_id:
            try:
                # Ambil komentar
                comments_data = db_manager.fetch_data("""
                    SELECT * FROM youtube_comments 
                    WHERE video_id = %s
                """, (selected_video_id,))
                
                # Detail video
                selected_video_data = data_video[data_video["video_id"] == selected_video_id].iloc[0]
                title = selected_video_data["title"]
                thumbnail_url = selected_video_data["thumbnail_url"]

                st.subheader("Video Detail")
                st.write(f"**Title**: {title}")
                st.image(thumbnail_url, width=300)

                if not comments_data.empty:
                    # Cek apakah prediksi sudah ada
                    predicted_data = db_manager.fetch_data("""
                        SELECT c.sentiment, p.predict_sentiment, c.comment
                        FROM predicted_sentiment p
                        JOIN youtube_comments c ON p.comment_id = c.comment_id
                        WHERE p.video_id = %s
                    """, (selected_video_id,))
                    
                    if not predicted_data.empty:
                        st.subheader("Predicted Sentiment")
                        st.dataframe(predicted_data)
                        
                        # Pie Chart distribusi prediksi
                        st.subheader("Predicted Sentiment Distribution")
                        sentiment_counts = predicted_data["predict_sentiment"].value_counts()
                        
                        labels = sentiment_counts.index
                        sizes = sentiment_counts.values
                        colors = ['#ff9999', '#66b3ff', '#99ff99']

                        fig, ax = plt.subplots()
                        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                        ax.axis('equal')
                        st.pyplot(fig)

                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = pd.crosstab(
                            predicted_data["sentiment"], 
                            predicted_data["predict_sentiment"]
                        )
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                        plt.xlabel('Predicted Sentiment')
                        plt.ylabel('True Sentiment')
                        st.pyplot(plt)

                    else:
                        # Tombol untuk melakukan analisis sentimen
                        if st.button("Analisis Sentiment"):
                            try:
                                # Ambil data preprocessed
                                preprocessed_data = preprocessor.preprocess_texts(comments_data['comment'])
                                
                                # Ambil data training yang sudah dipreprocess
                                training_data = db_manager.fetch_data("SELECT * FROM preprocessed_training")
                                
                                # Latih model
                                model_result = classifier.train(
                                    training_data['text'], 
                                    training_data['sentiment']
                                )
                                
                                # Prediksi
                                predictions = classifier.predict(preprocessed_data)
                                
                                # Simpan prediksi
                                predicted_df = pd.DataFrame({
                                    "sentiment": comments_data["sentiment"],
                                    "predict_sentiment": predictions,
                                    "comment": comments_data["comment"],
                                    "comment_id": comments_data["comment_id"],
                                    "video_id": comments_data["video_id"]
                                })
                                
                                # Simpan ke database
                                insert_prediction_query = """
                                    INSERT INTO predicted_sentiment 
                                    (predict_sentiment, comment_id, video_id) 
                                    VALUES (%s, %s, %s)
                                """
                                for _, row in predicted_df.iterrows():
                                    db_manager.execute_query(
                                        insert_prediction_query, 
                                        (row['predict_sentiment'], row['comment_id'], row['video_id'])
                                    )
                                
                                st.success(f"Analysis selesai! Akurasi: {model_result['accuracy']:.2%}")
                                st.subheader("Model Performance")
                                st.json(model_result['report'])
                            
                            except Exception as e:
                                st.error(f"Kesalahan analisis: {e}")
                                logger.error(f"Sentiment analysis error: {e}")
                else:
                    st.write("Tidak ada data untuk video ini.")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                logger.error(f"Sentiment analysis error: {e}")
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("KNN Sentiment Analysis © 2024")

if __name__ == "__main__":
    main()