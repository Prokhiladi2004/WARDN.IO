🛡️ WARDN.IO: Fraud Detection AI Pipeline

WARDN.IO is a production-ready, interactive FinTech Artificial Intelligence pipeline designed to detect, classify, and explain fraudulent transactions. Built with Streamlit, it provides a seamless end-to-end workflow from data ingestion to advanced SHAP-based explainability.

🌍 Live Demo: https://wardnio.streamlit.app/

🚀 Overview

Real-world fraud is rarely obvious; it requires reasoning across multiple fields simultaneously. WARDN.IO tackles this challenge by intelligently engineering context-aware features (velocity, sequence, and behavioral signals) and training a balanced Machine Learning model that maximizes the F1 score without sacrificing user experience.

Most importantly, WARDN.IO is not a black box. It utilizes SHAP (SHapley Additive exPlanations) to provide transaction-level transparency, revealing exactly why a specific transaction was flagged.

✨ Key Features

📂 Data Ingestion & Auto-Mocking: Seamlessly upload CSV or Parquet files. Missing critical FinTech columns are intelligently mocked for demonstration purposes.

🧹 Smart Data Cleaning: * Automatically merges legacy amt columns into a unified transaction_amount.

Quantifies exact counts of data quality issues (invalid IPs, missing amounts).

Builds a dynamic, data-driven City Normalization Lookup to resolve naming inconsistencies (e.g., "new york" → "New York").

⚙️ Advanced Feature Engineering:

Velocity Signals: Rolling 24-hour transaction counts and sums.

Sequence Signals: Calculates the exact time (in seconds) since the user's last transaction to catch bot-driven rapid fires.

Cross-User Signals: Analyzes device_account_ratio to detect device farms.

🤖 Robust ML Training: Uses a RandomForestClassifier with balanced class weights to handle extreme fraud imbalance. Evaluates using Precision, Recall, and F1-Score to reflect real-world business tradeoffs.

🧠 SHAP Explainability:

Global Importance: Understand the overarching features driving the model.

Local Explainability: Granular decision plots explaining the exact mathematical reasoning behind flagging an individual transaction.

🛠️ Tech Stack

Frontend/Framework: Streamlit

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-Learn (Random Forest)

Explainability: SHAP

Data Visualization: Matplotlib, Seaborn

💻 Installation & Local Setup

To run WARDN.IO locally, follow these steps:

Clone the repository:

git clone [https://github.com/yourusername/wardnio.git](https://github.com/yourusername/wardnio.git)
cd wardnio


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install dependencies:

pip install -r requirements.txt


(Note: Ensure your requirements.txt includes streamlit, pandas, numpy, scikit-learn, shap, matplotlib, and seaborn)

Run the application:

streamlit run app.py


📈 Fraud Patterns Detected

By leveraging our feature interactions and SHAP values, WARDN.IO is primed to detect sophisticated topologies such as:

The 'Test & Drain' Sequence: Micro-transactions followed by a massive spike relative to the user's historical median.

Device Farms: High device-to-account ratios indicating a single device cycling through stolen credentials.

Nocturnal ATO (Account Takeover): Spikes in unusual hours combined with high 24-hour velocity sums.

🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

Built with Python, Streamlit, and a passion for financial security.


![WhatsApp Image 2026-03-24 at 10 53 45 AM](https://github.com/user-attachments/assets/7e084633-7a37-4f44-9e5b-de3430204d0f)

