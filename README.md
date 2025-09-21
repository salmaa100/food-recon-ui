# 🍎 Food Reconciliation — UI + API

This project is a **food product reconciliation system** that connects user-provided data with the **OpenFoodFacts (OFF)** dataset.  
It was built as part of a final-year project for Goldsmiths University (Template 2: Data Reconciliation).

The app provides:
- **Single product search** with fuzzy matching
- **Batch reconciliation** via multi-line input
- **CSV upload/download** with cleaning log
- **OpenRefine-compatible API** (`/reconcile` endpoint)
- **Accessibility-friendly web UI** with dark/light theme

---

## 🚀 Features

- Real-time queries to OFF API  
- Fuzzy string matching (RapidFuzz) with brand weighting  
- Adjustable thresholds and Top-N results (5–30)  
- JSON API for interoperability  
- Supports OpenRefine as a reconciliation backend  

---

## 🛠️ Installation

Clone the repo:

```bash
git clone https://github.com/salmaa100/food-recon-ui.git
cd food-recon-ui

Create a virtual environment and install dependencies:
python3 -m venv venv
source venv/bin/activate   # on Mac/Linux
venv\Scripts\activate      # on Windows

pip install -r requirements.txt
Run the app:

uvicorn app:app --reload
Open http://127.0.0.1:8000 in your browser.

```
## Repository Structure

food-recon-ui/
│── app.py               # Main FastAPI app
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
│── LICENSE              # License (MIT)
│── .gitignore           # Git ignore file
│── sample.csv           # Example test dataset
└── /docs
    └── screenshots/     # Project screenshots

## 📊 Example Screenshots
Homepage

<img width="1378" height="877" alt="Screenshot 2025-09-21 005742" src="https://github.com/user-attachments/assets/b9af181d-e33e-4fde-9945-8cf636bb80c4" />


🔗 API Usage

The app exposes an OpenRefine-compatible Reconciliation API:
POST /reconcile
Content-Type: application/json

{
  "queries": {
    "q0": { "query": "milk" }
  }
}

Example JSON response:
{
  "q0": {
    "result": [
      {"id": "12345", "name": "Whole Milk", "score": 0.82, "match": true, "type": ["product"]}
    ]
  }
}

## 📜License

This project is licensed under the MIT License. See the LICENSE
 file for details.

 

## 🧑‍🎓Acknowledgements

OpenFoodFacts for their open dataset

FastAPI for the web framework

RapidFuzz for string matching

OpenRefine for the reconciliation standard




