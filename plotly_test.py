import plotly
from plotly.graph_objs import *

import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

ts=0.0
te=5.0
fz=1.0
A=1.0
sampling=1000  # Hz

t=np.arange(ts, te, 1/sampling)

Desired= A*np.sin(np.pi*fz*t)
noise = np.random.normal(0,0.02,len(t))
Measured= 0.90*A*np.sin(np.pi*fz*t - 0.1*np.pi) + noise


# Create traces
trace0 = go.Scatter(
    x = t,
    y = Desired,
    #mode = 'lines',
    name = 'Desired',
    line = dict(
        #color=('rgb(205, 12, 24)'),
        color='blue',
        width=3,
        dash='dash')
)
trace1 = go.Scatter(
    x = t,
    y = Measured,
    name = 'Measured',
    #mode='lines',
    line=dict(
        # color=('rgb(205, 12, 24)'),
        color='red',
        width=1,
        dash='line')
)
data = [trace0,trace1]

plotly.offline.plot({
    "data": data,
    "layout": Layout(title="Desired & Measured",
                     xaxis=XAxis(gridcolor='rgb(190,190,190)',showgrid=True),
                     yaxis=YAxis(gridcolor='rgb(190,190,190)', showgrid=True))
})