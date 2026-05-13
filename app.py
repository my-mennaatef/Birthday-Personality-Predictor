import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("birthday_data.csv")

# encoding
le_party = LabelEncoder()
le_activity = LabelEncoder()
le_color = LabelEncoder()
le_personality = LabelEncoder()

df["likes_party"] = le_party.fit_transform(df["likes_party"])

df["favorite_activity"] = le_activity.fit_transform(
    df["favorite_activity"]
)

df["favorite_color"] = le_color.fit_transform(
    df["favorite_color"]
)

df["personality"] = le_personality.fit_transform(
    df["personality"]
)

# table model

X = df.drop("personality", axis=1)

y = df["personality"]

model = DecisionTreeClassifier()

model.fit(X, y)

# page design

st.title("🎂 Birthday Personality Predictor")

st.write(
    "Discover your birthday personality using AI "
)

# userinput
age = st.slider(
    "Choose your age",
    15,
    40
)

likes_party = st.selectbox(
    "Do you like parties?",
    ["yes", "no"]
)

activity = st.selectbox(
    "Favorite activity",
    [
        "gaming",
        "reading",
        "travel",
        "movies",
        "dancing",
        "painting",
        "sports",
        "writing",
        "music"
    ]
)

color = st.selectbox(
    "Favorite color",
    [
        "black",
        "blue",
        "pink",
        "gray",
        "red",
        "purple",
        "green",
        "white",
        "yellow"
    ]
)
# perdiction

if st.button("Predict Personality"):

    input_data = pd.DataFrame(
        [[
            age,
            le_party.transform([likes_party])[0],
            le_activity.transform([activity])[0],
            le_color.transform([color])[0]
        ]],
        columns=[
            "age",
            "likes_party",
            "favorite_activity",
            "favorite_color"
        ]
    )

    prediction = model.predict(input_data)

    result = le_personality.inverse_transform(
        prediction
    )

    st.success(
        f" Your personality is: {result[0]}"
    )
    st.balloons()