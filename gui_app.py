
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("Agg")  # off-screen rendering for saved plots; GUI uses its own canvas below
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_loader import DATASETS
from models import run_experiment, HAS_XGBOOST

DATASET_LABELS = {
    "breast_cancer": "Breast Cancer (real UCI/sklearn data)",
    "heart_disease": "Heart Disease (synthetic, UCI schema)",
    "diabetes": "Diabetes (synthetic, UCI schema)",
}

# Human-friendly (label, help text) per feature, keyed by dataset
FEATURE_HINTS = {
    "heart_disease": {
        "age": "Age in years",
        "sex": "1 = male, 0 = female",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar > 120 mg/dl? 1=yes, 0=no",
        "restecg": "Resting ECG result (0-2)",
        "thalach": "Max heart rate achieved",
        "exang": "Exercise induced angina? 1=yes, 0=no",
        "oldpeak": "ST depression induced by exercise",
        "slope": "Slope of peak exercise ST segment (0-2)",
        "ca": "Number of major vessels colored (0-3)",
        "thal": "Thalassemia (1=normal, 2=fixed defect, 3=reversible)",
    },
    "diabetes": {
        "Pregnancies": "Number of pregnancies",
        "Glucose": "Plasma glucose concentration",
        "BloodPressure": "Diastolic blood pressure (mm Hg)",
        "SkinThickness": "Triceps skin fold thickness (mm)",
        "Insulin": "2-Hour serum insulin (mu U/ml)",
        "BMI": "Body mass index",
        "DiabetesPedigreeFunction": "Diabetes pedigree function score",
        "Age": "Age in years",
    },
}


class DiseasePredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Disease Prediction Toolkit")
        self.geometry("980x680")
        self.minsize(860, 600)

        # per-dataset cache: {dataset_key: {"X":.., "results_df":.., "fitted":.., "feature_defaults":..}}
        self.trained = {}

        self._build_layout()

    # ------------------------------------------------------------------ UI
    def _build_layout(self):
        header = ttk.Frame(self, padding=10)
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Disease Prediction from Medical Data",
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left")
        boost_note = "XGBoost available" if HAS_XGBOOST else "XGBoost not installed — using HistGradientBoosting fallback"
        ttk.Label(header, text=boost_note, foreground="#666").pack(side="right")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.train_tab = ttk.Frame(self.notebook)
        self.predict_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="1. Train & Compare")
        self.notebook.add(self.predict_tab, text="2. Predict a Patient")

        self._build_train_tab()
        self._build_predict_tab()

    # ---------------------------------------------------------- Train tab
    def _build_train_tab(self):
        top = ttk.Frame(self.train_tab, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Dataset:").pack(side="left", padx=(0, 6))
        self.train_dataset_var = tk.StringVar(value="breast_cancer")
        combo = ttk.Combobox(
            top,
            textvariable=self.train_dataset_var,
            values=list(DATASETS.keys()),
            state="readonly",
            width=18,
        )
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e: self._update_dataset_note())

        self.train_button = ttk.Button(top, text="Train All 4 Models", command=self._on_train_clicked)
        self.train_button.pack(side="left", padx=12)

        self.progress = ttk.Progressbar(top, mode="indeterminate", length=150)
        self.progress.pack(side="left", padx=6)

        self.dataset_note = ttk.Label(top, text="", foreground="#555")
        self.dataset_note.pack(side="left", padx=10)
        self._update_dataset_note()

        # Results table
        table_frame = ttk.Frame(self.train_tab, padding=(10, 0))
        table_frame.pack(fill="x")

        columns = ("Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=130 if col == "Model" else 100)
        self.tree.pack(fill="x")

        self.best_model_label = ttk.Label(self.train_tab, text="", font=("Segoe UI", 11, "bold"), padding=10)
        self.best_model_label.pack(anchor="w")

        # Plot canvas
        self.plot_frame = ttk.Frame(self.train_tab, padding=10)
        self.plot_frame.pack(fill="both", expand=True)
        self.canvas_widget = None

    def _update_dataset_note(self):
        key = self.train_dataset_var.get()
        self.dataset_note.config(text=DATASET_LABELS.get(key, ""))

    def _on_train_clicked(self):
        self.train_button.config(state="disabled")
        self.progress.start(10)
        thread = threading.Thread(target=self._train_worker, daemon=True)
        thread.start()

    def _train_worker(self):
        key = self.train_dataset_var.get()
        try:
            X, y = DATASETS[key]["loader"]()
            pretty_name = key.replace("_", " ").title()
            results_df, fitted = run_experiment(X, y, pretty_name)

            self.trained[key] = {
                "X": X,
                "y": y,
                "results_df": results_df,
                "fitted": fitted,
            }
            self.after(0, lambda: self._on_train_done(key, results_df, fitted))
        except Exception as exc:  # surface errors instead of freezing the UI
            self.after(0, lambda: self._on_train_error(exc))

    def _on_train_error(self, exc):
        self.progress.stop()
        self.train_button.config(state="normal")
        messagebox.showerror("Training failed", str(exc))

    def _on_train_done(self, key, results_df, fitted):
        self.progress.stop()
        self.train_button.config(state="normal")

        # populate table
        for row in self.tree.get_children():
            self.tree.delete(row)
        for _, row in results_df.iterrows():
            self.tree.insert(
                "",
                "end",
                values=(
                    row["Model"],
                    f"{row['Accuracy']:.4f}",
                    f"{row['Precision']:.4f}",
                    f"{row['Recall']:.4f}",
                    f"{row['F1-Score']:.4f}",
                    f"{row['ROC-AUC']:.4f}",
                ),
            )

        best = results_df.iloc[0]
        self.best_model_label.config(
            text=f"Best model: {best['Model']}  (ROC-AUC = {best['ROC-AUC']:.4f})"
        )

        self._render_comparison_plot(results_df, key)
        self._refresh_predict_tab_options()

    def _render_comparison_plot(self, results_df, dataset_key):
        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        x = range(len(results_df))
        width = 0.15
        for i, metric in enumerate(metrics):
            ax.bar([p + i * width for p in x], results_df[metric], width, label=metric)
        ax.set_xticks([p + width * 2 for p in x])
        ax.set_xticklabels(results_df["Model"], rotation=10, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=5, loc="lower center")
        ax.set_title(f"Model comparison — {dataset_key.replace('_', ' ').title()}")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_widget = canvas
        plt.close(fig)

    # -------------------------------------------------------- Predict tab
    def _build_predict_tab(self):
        top = ttk.Frame(self.predict_tab, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Dataset:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.predict_dataset_var = tk.StringVar(value="")
        self.predict_dataset_combo = ttk.Combobox(
            top, textvariable=self.predict_dataset_var, state="readonly", width=18, values=[]
        )
        self.predict_dataset_combo.grid(row=0, column=1, sticky="w")
        self.predict_dataset_combo.bind("<<ComboboxSelected>>", lambda e: self._build_patient_form())

        ttk.Label(top, text="Model:").grid(row=0, column=2, sticky="w", padx=(20, 6))
        self.predict_model_var = tk.StringVar(value="")
        self.predict_model_combo = ttk.Combobox(
            top, textvariable=self.predict_model_var, state="readonly", width=22, values=[]
        )
        self.predict_model_combo.grid(row=0, column=3, sticky="w")

        self.no_trained_label = ttk.Label(
            self.predict_tab,
            text="Train a dataset in Tab 1 first — it will then appear here.",
            foreground="#888",
            padding=20,
        )
        self.no_trained_label.pack()

        # Scrollable form area
        self.form_container = ttk.Frame(self.predict_tab, padding=10)
        canvas = tk.Canvas(self.form_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.form_container, orient="vertical", command=canvas.yview)
        self.form_inner = ttk.Frame(canvas)
        self.form_inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.form_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._predict_canvas = canvas

        bottom = ttk.Frame(self.predict_tab, padding=10)
        bottom.pack(fill="x")
        self.predict_button = ttk.Button(bottom, text="Predict", command=self._on_predict_clicked)
        self.predict_button.pack(side="left")
        self.fill_sample_button = ttk.Button(
            bottom, text="Fill with sample patient", command=self._fill_sample_values
        )
        self.fill_sample_button.pack(side="left", padx=10)

        self.result_label = ttk.Label(
            self.predict_tab, text="", font=("Segoe UI", 13, "bold"), padding=15
        )
        self.result_label.pack()

        self.entries = {}  # feature_name -> tk.Entry

    def _refresh_predict_tab_options(self):
        keys = list(self.trained.keys())
        self.predict_dataset_combo.config(values=keys)
        if keys and not self.predict_dataset_var.get():
            self.predict_dataset_var.set(keys[0])
        if keys:
            self.no_trained_label.pack_forget()
            self.form_container.pack(fill="both", expand=True)
            self._build_patient_form()

    def _build_patient_form(self):
        key = self.predict_dataset_var.get()
        if key not in self.trained:
            return

        # populate model options for this dataset
        model_names = list(self.trained[key]["fitted"].keys())
        self.predict_model_combo.config(values=model_names)
        if model_names:
            self.predict_model_var.set(model_names[0])

        # clear old form
        for child in self.form_inner.winfo_children():
            child.destroy()
        self.entries = {}

        X = self.trained[key]["X"]
        hints = FEATURE_HINTS.get(key, {})

        for i, col in enumerate(X.columns):
            ttk.Label(self.form_inner, text=col, width=26, anchor="w").grid(
                row=i, column=0, sticky="w", padx=5, pady=3
            )
            entry = ttk.Entry(self.form_inner, width=12)
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=3)
            hint = hints.get(col, "")
            if hint:
                ttk.Label(self.form_inner, text=hint, foreground="#888").grid(
                    row=i, column=2, sticky="w", padx=8
                )
            self.entries[col] = entry

        self.result_label.config(text="")

    def _fill_sample_values(self):
        key = self.predict_dataset_var.get()
        if key not in self.trained:
            return
        X = self.trained[key]["X"]
        sample = X.iloc[0]
        for col, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(sample[col]))

    def _on_predict_clicked(self):
        key = self.predict_dataset_var.get()
        model_name = self.predict_model_var.get()
        if key not in self.trained or not model_name:
            messagebox.showwarning("Not ready", "Train a dataset first, then pick a model.")
            return

        X = self.trained[key]["X"]
        try:
            values = {}
            for col in X.columns:
                raw = self.entries[col].get().strip()
                if raw == "":
                    raise ValueError(f"Missing value for '{col}'")
                values[col] = float(raw)
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        import pandas as pd

        row_df = pd.DataFrame([values], columns=X.columns)
        pipe = self.trained[key]["fitted"][model_name]["pipeline"]
        pred = pipe.predict(row_df)[0]
        proba = pipe.predict_proba(row_df)[0][1]

        positive_label = {
            "breast_cancer": ("Benign", "Malignant"),  # sklearn: 1=benign,0=malignant
            "heart_disease": ("No heart disease", "Heart disease present"),
            "diabetes": ("No diabetes", "Diabetes present"),
        }
        neg_text, pos_text = positive_label.get(key, ("Negative", "Positive"))
        # NOTE: for breast_cancer sklearn's target=1 is "benign" (negative finding),
        # for heart_disease/diabetes target=1 means disease present. Handle breast cancer specially.
        if key == "breast_cancer":
            outcome = neg_text if pred == 1 else pos_text
            risk_proba = 1 - proba if pred == 1 else proba
        else:
            outcome = pos_text if pred == 1 else neg_text
            risk_proba = proba if pred == 1 else 1 - proba

        color = "#b00020" if ("present" in outcome.lower() or outcome == "Malignant") else "#1a7a1a"
        self.result_label.config(
            text=f"Prediction: {outcome}   (confidence: {risk_proba:.1%})", foreground=color
        )


if __name__ == "__main__":
    app = DiseasePredictionApp()
    app.mainloop()
