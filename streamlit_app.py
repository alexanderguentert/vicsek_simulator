import vicsek
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

import streamlit as st

gif_name = "model.gif"

def make_gif(x, length_tail, frames):
    """
    Make GIF from Simulation Data x.
    frames is the number of Timesteps
    length_tail specifies the length of the tail (the point from former frames)
    """
    data = pd.DataFrame(x[length_tail])
    # plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    scat = ax.scatter(data[0], data[1],c='#1f77b4')
    plt.axis('square')

    alpha_frac = 1/length_tail
    hist = []
    for i in range(length_tail):
        data_i = pd.DataFrame(x[length_tail-i])
        scat_i = ax.scatter(data_i[0], data_i[1],c='#1f77b4', s=9,alpha=1-alpha_frac*i)
        hist.append(scat_i)

    def update(frame):
        data = pd.DataFrame(x[frame+length_tail])
        #scat = ax.scatter(data[0], data[1], c='blue')
        scat.set_offsets(data)

        for j in range(len(hist)):
            data_j =pd.DataFrame(x[frame+length_tail-j])
            hist[j].set_offsets(data_j)
        #return scat,scat2
    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames-length_tail, interval=30)

    writer = PillowWriter(fps=20)
    ani.save(gif_name, writer=writer)
    return ani




st.set_page_config('Vicsek Model Simulator')
st.title('Vicsek Model Simulator')
st.markdown(
    """
    The [Vicsek Model](https://en.wikipedia.org/wiki/Vicsek_model) is mathematical model that can be used to describe moving particles. 
    It is of relevance for many sciences like physics, biology, ecology or social sciences, as it can be used to model particles, animal swarms or even people flows.
    This simulator uses the [minimal implementation of the Vicsek](https://gist.github.com/arshednabeel/a70cc117eb38899fdd138f48b0bc5cd2) by [Arshed Nabeel](https://gist.github.com/arshednabeel).
    You can play with the parameters to see how they affect the model.
    """)
with st.sidebar:
    st.markdown('__Parameters:__')
    N = st.slider("Number of individuals", min_value=1, max_value=1000, value=100, step=None,)
    L = st.slider("Size of the domain", min_value=1, max_value=100, value=10, step=None,)
    R = st.slider("Interaction radius", min_value=0.1, max_value=10.0, value=1.0, step=None,)
    eta = st.slider("Noise level", min_value=0.01, max_value=1.0, value=0.1, step=None,)
    v = st.slider("Individual speed", min_value=0.1, max_value=10.0, value=1.0, step=None,)
    dt = st.slider("Time step", min_value=0.01, max_value=1.0, value=0.1, step=None,)
    T = st.slider("Total timesteps", min_value=10, max_value=300, value=30, step=None,)
    length_tail = st.slider("Tail length", min_value=1, max_value=10, value=5, step=None,)


with st.spinner("Loading..."):
    x, e = vicsek.simulate_vicsek_model(
        N = N,     # Number of individuals
        L = L,      # Size of the domain
        R = R,       # Interaction radius
        eta = eta,   # Noise level
        v = v,       # Individual speed
        dt = dt,    # Time step
        T = T,    # Total timesteps
    )
    ani = make_gif(x, length_tail, T)
    st.image(gif_name)

st.markdown('''Created by: Alexander Güntert 
            ([Mastodon](https://mastodon.social/@gntert), [Twitter](https://twitter.com/TrickTheTurner))  
            View source code: https://github.com/alexanderguentert/vicsek_simulator''')