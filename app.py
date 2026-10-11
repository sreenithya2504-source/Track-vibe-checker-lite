
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Track Vibe Checker",
    page_icon="🎵",
    layout="wide"
)

DATA_FILE = "spotify-tracks-dataset-detailed.csv"
MODEL_FILE = "popularity_model_lite.pkl"

FEATURES = [
    "danceability", "energy", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "loudness", "duration_ms"
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_tracks():
    columns = [
        "track_name", "artists", "popularity",
        "track_genre"
    ] + FEATURES

    df = pd.read_csv(DATA_FILE, usecols=columns)
    df = df.dropna(subset=["track_name", "artists"])
    return df


model = load_model()
tracks = load_tracks()


st.title("🎵 Track Vibe Checker")
st.markdown(
    "Discover your music's **popularity, audio personality, "
    "and age-group listening trends**."
)
st.caption("MUSIC ANALYTICS  •  MACHINE LEARNING  •  TRACK DISCOVERY")

st.subheader("🔎 Discover a Track")
st.write("Search the dataset to explore a song's popularity and audio profile.")

search = st.text_input(
    "Search for a track",
    placeholder="Enter a song title"
)

if search.strip():
    matches = tracks[
        tracks["track_name"].str.contains(
            search.strip(), case=False, regex=False, na=False
        )
    ]

    if matches.empty:
        st.warning("No matching track found. Try another title.")
    else:
        choices = (
            matches["track_name"] + " — " + matches["artists"]
        ).tolist()

        selected = st.selectbox("Choose a track", choices)
        index = choices.index(selected)
        track = matches.iloc[index]

        values = track[FEATURES].astype(float).to_frame().T
        predicted = float(model.predict(values)[0])
        predicted = max(0, min(100, predicted))

        st.subheader(f"{track['track_name']} — {track['artists']}")

        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="🎧 Dataset Popularity",
                value=f"{int(track['popularity'])}/100"
                )

        with col2:
            st.metric(
                label="🤖 ML-Predicted Popularity",
                value=f"{predicted:.1f}/100"
            )


        st.caption(
            "The model estimates popularity from audio features. "
            "This is not a live Spotify popularity lookup."
        )

        radar_features = [
            "danceability", "energy", "speechiness",
            "acousticness", "instrumentalness",
            "liveness", "valence"
        ]

        radar_values = [
            float(track[f]) for f in radar_features
        ]

        fig = go.Figure(
            go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_features + [radar_features[0]],
                fill="toself",
                name="Audio profile"
            )
        )

        fig.update_layout(
            title="Track Vibe Radar",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        
        st.subheader("🎼 Track Details")

        detail1, detail2 = st.columns(2)

        with detail1:
            st.metric("🎸 Genre", str(track["track_genre"]).title())

        with detail2:
            st.metric("🥁 Tempo", f"{track['tempo']:.1f} BPM")

st.divider()

st.header("👥 Explore Music by Age Group")
st.write(
    "Explore the most-listened tracks recorded for each age group "
    "in the age-specific listening dataset."
)
st.caption(
    "Note: Listening counts reflect the supplied dataset and may not "
    "represent the preferences of every person in an age group."
)

AGE_FILE = "top_tracks_by_age.csv"

@st.cache_data
def load_age_data():
    age_df = pd.read_csv(AGE_FILE)
    age_df["listens"] = pd.to_numeric(
        age_df["listens"], errors="coerce"
    )
    age_df = age_df.dropna(
        subset=["age_group", "artist", "track", "listens"]
    )
    return age_df

age_data = load_age_data()

age_groups = [
    "Kids (5-12)",
    "Young (13-17)",
    "Adult (18-59)",
    "Elder (60-80)"
]

available_groups = [
    group for group in age_groups
    if group in age_data["age_group"].unique()
]

selected_age = st.selectbox(
    "Select an age group",
    available_groups
)

age_subset = age_data[
    age_data["age_group"] == selected_age
].sort_values("listens", ascending=False)

if not age_subset.empty:
    st.subheader(f"📊 Top 10 Tracks for {selected_age}")

    top_tracks = age_subset.head(10).copy()
    top_tracks["song"] = (
        top_tracks["track"] + " — " + top_tracks["artist"]
    )

    st.bar_chart(
        top_tracks.set_index("song")["listens"],
        horizontal=True
    )

    st.dataframe(
        age_subset[["track", "artist", "listens"]].head(10),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No listening records are available for this age group.")

st.divider()

st.caption(
    "🎵 Track Vibe Checker Lite | Built with Streamlit, "
    "Pandas, Plotly & Scikit-learn"
)
