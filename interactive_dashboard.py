# pip install dash dash-bootstrap-components pandas networkx plotly

import dash
from dash import dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import os

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Step 1: Define the list of dates
dates = ['2025-02-03', '2025-02-04', '2025-02-05']

# Preprocess data and prepare figures
graphs = {}
edge_traces = {}
pie_charts = {}
suspicious_counts = {}

# Preparing the combined graph for consistent node positions
combined_graph = nx.Graph()

# Function to load and process data
def load_process_data():
    global graphs, edge_traces, pie_charts, suspicious_counts, combined_graph, node_positions
    graphs = {}
    edge_traces = {}
    pie_charts = {}
    suspicious_counts = {}
    combined_graph = nx.Graph()

    for date in dates:
        filename = f'Transactions_{date}.csv'

        # Check if the file exists
        if not os.path.isfile(filename):
            continue

        df = pd.read_csv(filename)

        # Ensure accounts are strings
        df['FromAccount'] = df['FromAccount'].astype(str)
        df['ToAccount'] = df['ToAccount'].astype(str)

        # Create Directed Graph
        G = nx.DiGraph()

        # Add nodes
        clients = set(df['FromAccount']).union(set(df['ToAccount']))
        G.add_nodes_from(clients)

        # Add edges
        for idx, row in df.iterrows():
            from_client = row['FromAccount']
            to_client = row['ToAccount']
            amount = row['Amount']
            payment_type = row['PaymentType']
            suspicious = row['Suspicious']

            edge_attrs = {
                'amount': amount,
                'payment_type': payment_type,
                'suspicious': suspicious
            }

            G.add_edge(from_client, to_client, **edge_attrs)

        # Store the graph
        graphs[date] = G

        # Compute suspicious and regular operation counts
        suspicious_count = df[df['Suspicious'] == 'Yes'].shape[0]
        regular_count = df[df['Suspicious'] == 'No'].shape[0]
        suspicious_counts[date] = {
            'Suspicious': suspicious_count,
            'Regular': regular_count
        }

        # Add to combined graph for node positions
        combined_graph.add_nodes_from(G.nodes())
        combined_graph.add_edges_from(G.edges())

    # Use spring layout in 3D to compute positions
    pos = nx.spring_layout(combined_graph, dim=3, k=0.5, iterations=100, seed=42)

    # Extract node positions
    node_positions = {}
    for node in combined_graph.nodes():
        node_positions[node] = pos[node]

    # Prepare edge traces for each date
    for date in dates:
        if date not in graphs:
            continue
        G = graphs[date]
        edge_trace_suspicious = dict(
            x=[], y=[], z=[], mode='lines',
            line=dict(color='red', width=4), hoverinfo='text', text=[]
        )
        edge_trace_normal = dict(
            x=[], y=[], z=[], mode='lines',
            line=dict(color='gray', width=2), hoverinfo='text', text=[]
        )
        for edge in G.edges(data=True):
            x0, y0, z0 = node_positions.get(edge[0], (0, 0, 0))
            x1, y1, z1 = node_positions.get(edge[1], (0, 0, 0))
            edge_info = f"From: {edge[0]}<br>To: {edge[1]}<br>Amount: ${edge[2]['amount']:.2f}<br>Type: {edge[2]['payment_type']}"
            if edge[2]['suspicious'] == 'Yes':
                edge_trace = edge_trace_suspicious
            else:
                edge_trace = edge_trace_normal
            # Add positions
            edge_trace['x'] += [x0, x1, None]
            edge_trace['y'] += [y0, y1, None]
            edge_trace['z'] += [z0, z1, None]
            # Add hover text
            edge_trace['text'] += [edge_info, edge_info, None]
        # Store edge traces
        edge_traces[date] = {'normal': edge_trace_normal, 'suspicious': edge_trace_suspicious}

        # Prepare the pie chart for the date
        counts = suspicious_counts[date]
        labels = ['Regular Operations', 'Suspicious Operations']
        values = [counts['Regular'], counts['Suspicious']]
        pie = go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=['gray', 'red']),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )
        pie_charts[date] = pie

    return node_positions

# Function to prepare data for the stacked bar chart
def prepare_stacked_bar_data(selected_date, top_n=5):
    # Load data for the selected date
    filename = f'Transactions_{selected_date}.csv'
    df = pd.read_csv(filename)

    # Ensure accounts are strings
    df['FromAccount'] = df['FromAccount'].astype(str)
    df['ToAccount'] = df['ToAccount'].astype(str)

    # Filter for suspicious transactions
    df_suspicious = df[df['Suspicious'] == 'Yes']

    # Check if there are any suspicious transactions
    if df_suspicious.empty:
        return pd.DataFrame(columns=['Client', 'Counterparty', 'Amount'])

    # Combine senders and receivers
    df_from = df_suspicious[['FromAccount', 'ToAccount', 'Amount']].rename(
        columns={'FromAccount': 'Client', 'ToAccount': 'Counterparty'}
    )
    df_to = df_suspicious[['ToAccount', 'FromAccount', 'Amount']].rename(
        columns={'ToAccount': 'Client', 'FromAccount': 'Counterparty'}
    )
    df_combined = pd.concat([df_from, df_to], ignore_index=True)

    # Aggregate total amount per client and counterparty
    df_agg = df_combined.groupby(['Client', 'Counterparty']).agg({'Amount': 'sum'}).reset_index()

    # For each client, limit to top N counterparties
    aggregated_list = []
    clients = df_agg['Client'].unique()
    for client in clients:
        df_client = df_agg[df_agg['Client'] == client].copy()
        # Sort by amount
        df_client = df_client.sort_values('Amount', ascending=False)
        # Separate top N counterparties
        top_df = df_client.head(top_n)
        others_df = df_client.iloc[top_n:]
        # Sum up 'Others'
        if not others_df.empty:
            others_total = others_df['Amount'].sum()
            other_row = pd.DataFrame({
                'Client': [client],
                'Counterparty': ['Others'],
                'Amount': [others_total]
            })
            top_df = pd.concat([top_df, other_row], ignore_index=True)
        aggregated_list.append(top_df)

    aggregated_data = pd.concat(aggregated_list, ignore_index=True)

    return aggregated_data

# Function to create the stacked bar chart
def create_stacked_bar_chart(aggregated_data):
    if aggregated_data.empty:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No suspicious transactions for the selected date.",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16),
            x=0.5,
            y=0.5,
            xanchor='center',
            yanchor='middle'
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            margin=dict(l=0, r=0, b=0, t=0)
        )
        return fig

    # Create the stacked bar chart
    fig = px.bar(
        aggregated_data,
        x='Client',
        y='Amount',
        color='Counterparty',
        title='Clients with Suspicious Transactions',
        labels={
            'Amount': 'Total Amount',
            'Client': 'Client',
            'Counterparty': 'Counterparty'
        }
    )

    # Update layout for better visuals
    fig.update_layout(
        barmode='stack',
        xaxis_title='Client',
        yaxis_title='Total Suspicious Amount Transacted',
        legend_title='Counterparty',
        margin=dict(l=10, r=10, b=10, t=30),  # Adjusted margins
        title_font_size=14,  # Reduced title font size
        title_y=0.95  # Adjusted title position
    )

    return fig

# Load and process data initially
node_positions = load_process_data()

# Step 2: Build the Dash layout
app.layout = dbc.Container(
    [
        html.H1("Transaction Network Dashboard", style={'textAlign': 'center', 'marginTop': 20}),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Date:"),
                                dcc.Dropdown(
                                    id='date-dropdown',
                                    options=[{'label': date, 'value': date} for date in dates],
                                    value=dates[0],
                                    clearable=False
                                ),
                            ],
                            width=3
                        ),
                        dbc.Col(
                            [
                                html.Button('Refresh Data', id='refresh-button', n_clicks=0),
                            ],
                            width=2
                        )
                    ],
                    align='center',
                    style={'marginBottom': 20}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id='network-graph', style={'height': '70vh'})
                            ],
                            width=8
                        ),
                        dbc.Col(
                            [
                                dcc.Graph(id='stacked-bar-chart', style={'height': '32vh'}),
                                html.Div(style={'height': '6vh'}),  # Spacer
                                dcc.Graph(id='pie-chart', style={'height': '32vh'})
                            ],
                            width=4
                        )
                    ]
                )
            ]
        )
    ],
    fluid=True
)

# Step 3: Define callback to update graphs
@app.callback(
    [Output('network-graph', 'figure'),
     Output('stacked-bar-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('date-dropdown', 'value'),
     Input('refresh-button', 'n_clicks')]
)
def update_graphs(selected_date, n_clicks):
    # Reload data if refresh button is clicked
    if n_clicks > 0:
        load_process_data()
        # Reset n_clicks to prevent continuous reloads
        n_clicks = 0

    # Node positions
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    for node in combined_graph.nodes():
        x, y, z = node_positions.get(node, (0, 0, 0))
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f'{node}')

    # Node trace
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(
            symbol='circle',
            size=5,
            color='blue',
        ),
        text=node_text,
        hoverinfo='text',
        name='Clients'
    )

    # Edge traces
    edge_traces_normal = edge_traces[selected_date]['normal']
    edge_traces_suspicious = edge_traces[selected_date]['suspicious']

    # Network graph figure
    fig = go.Figure()
    fig.add_trace(node_trace)
    fig.add_trace(go.Scatter3d(**edge_traces_normal, showlegend=False))
    fig.add_trace(go.Scatter3d(**edge_traces_suspicious, showlegend=False))
    fig.update_layout(
        title=f"3D Network Graph for {selected_date}",
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Prepare data for the stacked bar chart
    aggregated_data = prepare_stacked_bar_data(selected_date)

    # Create the stacked bar chart
    stacked_bar_fig = create_stacked_bar_chart(aggregated_data)

    # Pie chart figure
    pie_fig = go.Figure(pie_charts[selected_date])
    pie_fig.update_layout(
        title=f"Operations Breakdown for {selected_date}",
        margin=dict(l=10, r=10, b=30, t=30),  # Adjusted margins
        title_font_size=14,  # Reduced title font size
        title_y=0.05 #0.95  # Adjusted title position
    )

    return fig, stacked_bar_fig, pie_fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
