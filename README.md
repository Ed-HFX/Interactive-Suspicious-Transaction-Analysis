Transaction 3D Network Dashboard
Overview

This project is a Dash-based interactive dashboard for visualizing financial transaction networks using NetworkX and Plotly. It processes transaction data from CSV files, constructs a 3D network graph, and provides stacked bar charts and pie charts to analyze suspicious transactions.
Features

✅ 3D Transaction Network Graph – Visualizes relationships between clients using NetworkX and Plotly.
✅ Suspicious Transaction Analysis – Highlights suspicious transactions in red.
✅ Stacked Bar Chart – Shows the top counterparties in suspicious transactions.
✅ Pie Chart – Displays the proportion of regular vs. suspicious transactions.
✅ Interactive Dropdowns & Refresh Button – Users can select different dates and reload data.
Installation

Ensure you have Python installed, then run:

pip install dash dash-bootstrap-components pandas networkx plotly

Usage

    Place transaction CSV files (e.g., Transactions_2025-02-03.csv) in the project directory.
    Run the app:

    python interactive_dashboard.py

    Open the web browser and go to http://127.0.0.1:8050/

Data Format

Each CSV file should contain the following columns:
Timestamp	FromBank	FromAccount	ToAccount	Amount	PaymentType	Suspicious

Visualization Components

📌 Network Graph:

    Nodes = Clients
    Edges = Transactions
    Red edges = Suspicious transactions
    Gray edges = Regular transactions

📊 Stacked Bar Chart:

    Shows top counterparties involved in suspicious transactions.
    Aggregates amounts transacted.

🥧 Pie Chart:

    Breaks down regular vs. suspicious transactions for the selected date.


