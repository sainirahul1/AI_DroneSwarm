import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import time
import pandas as pd
from engine import DroneSwarmEnv, MADDPGAgent, ReplayBuffer

# Page config
st.set_page_config(page_title="Horizon | Drone Swarm AI Forge", layout="wide", page_icon="🛸")

# Custom CSS for Premium Horizon Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at top right, #1a1c2c, #0d0f17) !important;
        color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }

    /* Glassmorphism containers */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }
    
    .stSidebar {
        background: rgba(13, 15, 23, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .neon-text {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        font-weight: 800;
    }
    
    .hero-title {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #00d4ff, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        font-weight: 800;
    }
    
    .mission-card {
        background: rgba(0, 212, 255, 0.05);
        padding: 2.5rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin-bottom: 25px;
        box-shadow: inset 0 0 20px rgba(0, 212, 255, 0.05);
    }
    
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        padding: 8px 16px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 212, 255, 0.2) !important;
        border-bottom: 2px solid #00d4ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# App Content Starts
st.markdown('<h1 class="hero-title">HORIZON</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.2rem; margin-top: -10px; color: #aaa;">Autonomous Collective Intelligence Forge</p>', unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div class="mission-card">
        <span style="color: #00d4ff; font-weight: 800; font-size: 0.9rem; letter-spacing: 2px;">TACTICAL BRIEFING</span><br><br>
        Deploy an autonomous drone swarm to navigate dynamic terrain using <b>MADDPG Reinforcement Learning</b>. 
        Agents must synchronize movements to reach the target while maintaining safety protocols.
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="neon-text">SYSTEM CONFIG</h2>', unsafe_allow_html=True)
    n_agents = st.slider("Swarm Size", 2, 5, 3)
    world_size = st.slider("Operational Theater (Area)", 5, 20, 10)
    
    with st.expander("Advanced Brain Parameters"):
        batch_size = st.select_slider("Neural Batch Size", options=[32, 64, 128], value=64)
        episodes = st.number_input("Training Cycles", min_value=10, max_value=1000, value=100)
        steps_per_episode = st.number_input("Runtime Steps", min_value=10, max_value=500, value=100)
    
    st.markdown("---")
    if st.button("🗑️ Reset Neural Weights", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Environment Init
if 'env' not in st.session_state or st.session_state.n_agents != n_agents:
    st.session_state.n_agents = n_agents
    st.session_state.env = DroneSwarmEnv(n_agents=n_agents, world_size=world_size)
    
    # Calculate obs_dim based on new engine logic
    # obs = [own(2), others_rel(2*(n-1)), target_rel(2)]
    obs_dim = 2 + 2*(n_agents-1) + 2
    act_dim = 2
    
    st.session_state.agents = [
        MADDPGAgent(obs_dim, act_dim, obs_dim*n_agents, act_dim*n_agents)
        for _ in range(n_agents)
    ]
    st.session_state.buffer = ReplayBuffer()
    st.session_state.reward_history = []
    st.session_state.success_history = []

# UI Tabs
tab1, tab2, tab3 = st.tabs(["🛸 Tactical Simulation", "🧠 Cognitive Analytics", "📖 Operations Manual"])

with tab1:
    col_main, col_stat = st.columns([3, 1])
    
    with col_main:
        plot_placeholder = st.empty()
        progress_bar = st.progress(0)
        start_btn = st.button("🚀 INITIATE DEPLOYMENT", type="primary", use_container_width=True)
        
    with col_stat:
        st.markdown('<p class="neon-text">LIVE DATA</p>', unsafe_allow_html=True)
        m_reward = st.metric("Swarm Reward", "N/A")
        m_episode = st.metric("Cycles Sync", "0")
        m_success = st.metric("Mission Success", "0%")
        m_collision = st.metric("Safety Violations", "0")

with tab2:
    st.markdown('<p class="neon-text">LEARNING PROGRESS</p>', unsafe_allow_html=True)
    chart_placeholder = st.empty()

with tab3:
    st.markdown("""
    ### How the AI Works
    - **Perception**: Each drone senses its position, the target's relative vector, and other drones' locations.
    - **Inference**: The **Actor Network** (Brain) decides the optimal thrust vector.
    - **Optimization**: The **Critic Network** (Teacher) evaluates actions based on swarm proximity and safety.
    - **Reward Shaping**: Agents receive positive reinforcement for closing distance and heavy penalties for collisions.
    """)

# Simulation Engine
def update_plot(env, episode, step, collision_alert=False):
    fig = go.Figure()
    
    # Grid styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        xaxis=dict(range=[0, world_size], gridcolor="#222", zeroline=False),
        yaxis=dict(range=[0, world_size], gridcolor="#222", zeroline=False),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        showlegend=False
    )

    # Target
    fig.add_trace(go.Scatter(
        x=[env.target[0]], y=[env.target[1]],
        mode='markers',
        marker=dict(size=25, color='#F9D423', symbol='star', line=dict(width=2, color='white')),
        name='Target'
    ))

    # Drones
    colors = ['#00d4ff', '#ff007f', '#00ff41', '#ffaa00', '#aa00ff']
    for i in range(env.n_agents):
        # Trail (Simple version using current and last pos if we tracked it, but let's just do icon)
        fig.add_trace(go.Scatter(
            x=[env.positions[i, 0]], y=[env.positions[i, 1]],
            mode='markers+text',
            text=[f"Drone {i+1}"],
            textposition="top center",
            marker=dict(size=20, color=colors[i % len(colors)], symbol='triangle-up', 
                        line=dict(width=1, color='white')),
        ))
        
        # Collision Ring
        fig.add_shape(type="circle",
            x0=env.positions[i, 0]-0.3, y0=env.positions[i, 1]-0.3,
            x1=env.positions[i, 0]+0.3, y1=env.positions[i, 1]+0.3,
            line=dict(color=colors[i % len(colors)], width=1),
            fillcolor=colors[i % len(colors)], opacity=0.1
        )

    if collision_alert:
        fig.add_annotation(text="COLLISION DETECTED", x=world_size/2, y=world_size-1, 
                          showarrow=False, font=dict(size=20, color="red", family="Inter"))

    return fig

if start_btn:
    success_count = 0
    total_collisions = 0
    
    for episode in range(episodes):
        obs = st.session_state.env.reset()
        episode_reward = 0
        collision_this_ep = False
        
        for step in range(steps_per_episode):
            # Prediction
            obs_tensor = torch.FloatTensor(obs)
            actions = []
            for i in range(n_agents):
                with torch.no_grad():
                    action = st.session_state.agents[i].actor(obs_tensor[i]).numpy()
                # Exploration Noise
                noise = np.random.normal(0, 0.1 * (1 - episode/episodes), size=2)
                actions.append(action + noise)

            next_obs, rewards, done = st.session_state.env.step(actions)
            st.session_state.buffer.push(obs, actions, rewards, next_obs, done)
            obs = next_obs
            episode_reward += np.mean(rewards)

            # Detect collision for UI
            for i in range(n_agents):
                for j in range(i+1, n_agents):
                    if np.linalg.norm(st.session_state.env.positions[i] - st.session_state.env.positions[j]) < 0.6:
                        total_collisions += 1
                        collision_this_ep = True

            if step % 8 == 0:
                plot_placeholder.plotly_chart(update_plot(st.session_state.env, episode, step, collision_alert=collision_this_ep), 
                                            use_container_width=True, key=f"plot_{episode}_{step}")
            
            # Training Logic
            if len(st.session_state.buffer) > 1000:
                batch_obs, batch_acts, batch_rewards, batch_next_obs, batch_done = st.session_state.buffer.sample(batch_size)
                # (Training code remains same but integrated here)
                for i, agent in enumerate(st.session_state.agents):
                    obs_all = torch.FloatTensor(batch_obs.reshape(batch_size, -1))
                    acts_all = torch.FloatTensor(batch_acts.reshape(batch_size, -1))
                    next_obs_all = torch.FloatTensor(batch_next_obs.reshape(batch_size, -1))
                    
                    next_actions = []
                    for j in range(n_agents):
                        with torch.no_grad():
                            next_actions.append(st.session_state.agents[j].target_actor(torch.FloatTensor(batch_next_obs[:, j, :])))
                    next_acts_all = torch.cat(next_actions, dim=1)
                    
                    reward = torch.FloatTensor(batch_rewards[:, i]).unsqueeze(1)
                    done_mask = torch.FloatTensor(batch_done).unsqueeze(1)
                    with torch.no_grad():
                        target_q = reward + agent.gamma * (1 - done_mask) * agent.target_critic(next_obs_all, next_acts_all)
                    
                    current_q = agent.critic(obs_all, acts_all)
                    critic_loss = torch.nn.MSELoss()(current_q, target_q)
                    agent.critic_opt.zero_grad(); critic_loss.backward(); agent.critic_opt.step()
                    
                    curr_actions = []
                    for j in range(n_agents):
                        if j == i: curr_actions.append(agent.actor(torch.FloatTensor(batch_obs[:, j, :])))
                        else: curr_actions.append(torch.FloatTensor(batch_acts[:, j, :]))
                    curr_acts_all = torch.cat(curr_actions, dim=1)
                    actor_loss = -agent.critic(obs_all, curr_acts_all).mean()
                    agent.actor_opt.zero_grad(); actor_loss.backward(); agent.actor_opt.step()
                    agent.soft_update()

            if done:
                success_count += 1
                break
        
        # History
        st.session_state.reward_history.append(episode_reward)
        st.session_state.success_history.append((success_count / (episode + 1)) * 100)
        
        # Metrics Update
        m_reward.metric("Swarm Reward", f"{episode_reward:.1f}")
        m_episode.metric("Cycles Sync", f"{episode + 1}")
        m_success.metric("Mission Success", f"{(success_count / (episode + 1)) * 100:.1f}%")
        m_collision.metric("Safety Violations", f"{total_collisions}")
        progress_bar.progress((episode + 1) / episodes)
        
        # Chart Update
        if episode % 5 == 0:
            with tab2:
                df = pd.DataFrame({"Episode": range(len(st.session_state.reward_history)), 
                                  "Success Rate": st.session_state.success_history})
                
                # Using Plotly for the success chart to support keys and keep "Horizon" theme
                fig_success = go.Figure()
                fig_success.add_trace(go.Scatter(x=df["Episode"], y=df["Success Rate"], 
                                               mode='lines+markers', line=dict(color='#00d4ff', width=3),
                                               fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'))
                fig_success.update_layout(
                    template="plotly_dark",
                    title="Mission Success Progression",
                    xaxis_title="Training Episode", yaxis_title="Success Rate (%)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                    margin=dict(l=20, r=20, t=50, b=20), height=400
                )
                chart_placeholder.plotly_chart(fig_success, use_container_width=True, key=f"success_plot_{episode}")

st.success("Deployment Mission Complete!")
