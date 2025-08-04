import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import joblib
import re
import os


ctk.set_appearance_mode("dark") 
ctk.set_default_color_theme("blue") 

BASE_DIR = r"C:\Users\Rohit\OneDrive\Desktop\aitask"
MODEL_PATH = os.path.join(BASE_DIR, "naive_bayes_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    exit(1)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class FakeNewsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Fake News Detection AI")
        self.geometry("650x450")
        self.resizable(False, False)

     
        self.label_welcome = ctk.CTkLabel(
            self, 
            text="Welcome to Fake News Detection", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.label_welcome.pack(pady=(25, 15))

      
        self.input_text = ctk.CTkTextbox(
            self, 
            width=600, 
            height=120, 
            font=ctk.CTkFont(size=14)
        )
        self.input_text.pack(pady=(0, 20))

        
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=(0, 10))

        
        self.btn_predict = ctk.CTkButton(
            self.button_frame, 
            text="Predict", 
            command=self.predict_news, 
            width=130, height=40
        )
        self.btn_predict.grid(row=0, column=0, padx=20, pady=5)

        
        self.btn_clear = ctk.CTkButton(
            self.button_frame, 
            text="Clear", 
            command=self.clear_all, 
            width=130, height=40,
            fg_color="#888888",
            hover_color="#555555"
        )
        self.btn_clear.grid(row=0, column=1, padx=20, pady=5)

        
        self.btn_info = ctk.CTkButton(
            self.button_frame, 
            text="Show Model Info", 
            command=self.show_model_info, 
            width=150, height=35
        )
        self.btn_info.grid(row=0, column=2, padx=20, pady=5)

        
        self.result_label = ctk.CTkLabel(
            self, 
            text="", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.result_label.pack(pady=(15, 25))

        
        self.footer_label = ctk.CTkLabel(
            self, 
            text="Made by Rohit Shah | Module Leader: Suman Shrestha", 
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA"
        )
        self.footer_label.pack(side="bottom", pady=10)

    def predict_news(self):
        text = self.input_text.get("0.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter a news title to predict.")
            return

        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            self.result_label.configure(text="REAL NEWS", text_color="#00FF00") 
        else:
            self.result_label.configure(text="FAKE NEWS", text_color="#FF4444") 

    def clear_all(self):
        self.input_text.delete("0.0", tk.END)
        self.result_label.configure(text="")

    def show_model_info(self):
        info_text = (
            "Model: Multinomial Naive Bayes\n"
            "Vectorizer: CountVectorizer\n"
            "Trained on cleaned news title text\n"
            "Prediction output:\n 1 = Real News\n 0 = Fake News"
        )
        messagebox.showinfo("Model Info", info_text)

if __name__ == "__main__":
    app = FakeNewsApp()
    app.mainloop()
