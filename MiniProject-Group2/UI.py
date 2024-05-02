import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')  
from Main1 import run_prediction



def validate_date(date_text, start_date, end_date):
    try:
        datetime_obj = datetime.strptime(date_text, '%Y-%m-%d')
        if datetime_obj < start_date or datetime_obj > end_date:
            messagebox.showerror("Error", f"Date not in range of the Database, please enter a date between {start_date.date()} to {end_date.date()}")
            return False
        return True
    except ValueError:
        messagebox.showerror("Error", "Wrong format. Please enter again in YYYY-MM-DD format.")
        return False

def on_submit():
    start_date_str = start_date_entry.get()
    end_date_str = end_date_entry.get()
    if validate_date(start_date_str, dataset_start_date, dataset_end_date) and validate_date(end_date_str, dataset_start_date, dataset_end_date):
        
        run_prediction(start_date_str, end_date_str, r'/home/kartik/Desktop/Mini_Project_ML/AAPL.csv')  
        messagebox.showinfo("Success", "The prediction has completed successfully.")

root = tk.Tk()
root.title("Apple Stock Data Predictor")

dataset_start_date = datetime(2010, 11, 10)
dataset_end_date = datetime(2023, 11, 9)

tk.Label(root, text="Enter the start date (YYYY-MM-DD):").pack()
start_date_entry = tk.Entry(root)
start_date_entry.pack()

tk.Label(root, text="Enter the end date (YYYY-MM-DD):").pack()
end_date_entry = tk.Entry(root)
end_date_entry.pack()

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

root.mainloop()
