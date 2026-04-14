import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="🏪 Tom Nook’s Prediction Shop API",
    description=(
        "Welcome to **Tom Nook’s Prediction Shop!** 🦝✨\n\n"
        "An AI-powered service where island analytics meets business instincts.\n\n"
        "### 💼 Services Available:\n"
        "- ☕ **Nook’s Revenue Estimator** → Predict item resale value\n"
        "- 🐶 **Isabelle’s Decision Desk** → Will your gift be returned?\n"
        "- 🐟 **Resetti’s Fish Optimizer** → Estimate fish prices\n"
        "- 🐶 **Villager Classifier** → Predict villager traits\n"
        "- 🦉 **Blathers’ Gift Recommender** → Best gift suggestions\n"
        "- 🌟 **Fishing Decision Tree** → Is the catch worth it?\n\n"
        "### 🔮 Features:\n"
        "- 🤖 Machine Learning powered predictions\n"
        "- 🎮 Game-style interactive responses\n"
        "- ⚡ Fast and lightweight API\n"
    ),
    version="1.0.0",
)

origins = [
    # Local development
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    # GitHub Pages
    "https://c-akanksha.github.io",
    "https://c-akanksha.github.io/tom-nooks-prediction-shop",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD MODELS
# -----------------------------
nook_model = joblib.load("models/nooks_revenue_estimator.pkl")
isabelle_model = joblib.load("models/isabelles_decision_desk.pkl")
fish_model = joblib.load("models/resettis_fish_price_optimizer.pkl")
villager_classifier_model = joblib.load("models/resettis_villager_classifier.pkl")
gift_model = joblib.load("models/blathers_gift_recommendation_engine.pkl")
fishing_decision_model = joblib.load("models/villager_fishing_decision_tree.pkl")
encoders = joblib.load("models/encoders.pkl")

# -----------------------------
# INPUT SCHEMAS
# -----------------------------
class NookInput(BaseModel):
    buy_price: float

class IsabelleInput(BaseModel):
    sell_price: float

class FishInput(BaseModel):
    shadow_size: str
    location: str

class VillagerInput(BaseModel):
    hobby: str

class GiftInput(BaseModel):
    species: str
    personality: str
    color: str

class FishingDecisionInput(BaseModel):
    shadow: str
    location: str
    spawn_rate: float

# -----------------------------
# HELPERS
# -----------------------------
shadow_map = {
    "X-Small": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "X-Large": 5,
    "XX-Large": 6
}

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "🏪 Welcome to Tom Nook’s Prediction Shop!"}

# ☕ Nook
@app.post("/nook")
def predict_nook(data: NookInput):
    df = pd.DataFrame([[data.buy_price]], columns=["Buy_Num"])
    pred = float(nook_model.predict(df)[0])

    sell_price = round(pred, 2)

    if sell_price < 100:
        message = "🦝 'Oh dear… that’s not a great return, hm?'"
    elif sell_price < 1000:
        message = "🦝 'Hmm… modest gains, but every Bell counts!'"
    elif sell_price < 5000:
        message = "🦝 'Now that’s a decent deal! Well done!'"
    else:
        message = "🦝 'Excellent! That’s a profitable venture indeed!'"

    return {
        "sell_price": sell_price,
        "message": message
    }

# 🐶 Isabelle
@app.post("/isabelle")
def predict_isabelle(data: IsabelleInput):
    df = pd.DataFrame([[data.sell_price]], columns=["Sell"])
    prob = float(isabelle_model.predict_proba(df)[0][1])

    if prob < 0.2:
        message = "🐶 'Oh! That’s… thoughtful! I’m sure they appreciated the gesture… even if just a little!' 🙃"
        result = "🙃 Just thanks"
    elif prob < 0.5:
        message = "🐶 'Aww, that’s a nice gift! They might not give something back, but it’ll make them smile!' 🙂"
        result = "🙂 Maybe no gift"
    elif prob < 0.8:
        message = "🐶 'Ooo! That’s a pretty good gift! I think there’s a good chance they’ll return the favor!' 🎁"
        result = "🎁 Likely gift back"
    else:
        message = "🐶 'Wow!! That’s an AMAZING gift! I’m sure they’ll be absolutely delighted—and give you something great in return!' 🎉"
        result = "🎉 Definitely gift back"

    return {
        "probability": round(prob, 4),
        "result": result,
        "message": message
    }

# 🐟 Fish Optimizer
@app.post("/fish-price")
def predict_fish(data: FishInput):
    df = pd.DataFrame(0, index=[0], columns=fish_model.feature_names_in_)

    df["Shadow_Num"] = shadow_map.get(data.shadow_size, 3)

    loc_col = f"Loc_{data.location}"
    if loc_col in df.columns:
        df[loc_col] = 1

    pred = float(fish_model.predict(df)[0])
    price = round(pred, 2)

    if price < 500:
        message = "💢 'Are ya kiddin' me?! That fish ain't worth the bait you used!' 😡"
    elif price < 2000:
        message = "💢 'Hmph… I've seen better. Don't get cocky.' 😤"
    elif price < 5000:
        message = "💢 'Alright… not terrible. You're learning, I guess.' 😐"
    else:
        message = "💢 'Now THAT'S a catch!! Finally, something worth my time!' 😲"

    return {
        "predicted_price": price,
        "message": message
    }

@app.post("/villager-classifier")
def predict_villager(data: VillagerInput):
    try:
        hobby_clean = data.hobby.strip().title()
        hobby_id = encoders["Hobby"].transform([hobby_clean])[0]

        df = pd.DataFrame([[hobby_id]], columns=["Hobby_Num"])

        probs = villager_classifier_model.predict_proba(df)[0]
        max_idx = probs.argmax()
        confidence = probs[max_idx]

        pred_class = villager_classifier_model.classes_[max_idx]
        gender = encoders["Gender"].inverse_transform([pred_class])[0]

        if confidence < 0.6:
            message = "🐶 'Hmm… I’m not entirely sure, but I’ll give it my best guess!' 🤔"
        elif confidence < 0.85:
            message = "🐶 'Oh! I feel pretty confident about this one!' 😊"
        else:
            message = "🐶 'Hehe! I’m absolutely certain about this assessment!' 🌟"

        return {
            "prediction": gender,
            "confidence": round(confidence, 3),
            "message": message
        }

    except ValueError:
        return {
            "error": "🐶 'Oh no! I couldn’t recognize that hobby. Could you try another one?'"
        }

@app.post("/gift-recommendation")
def predict_gift(data: GiftInput):
    try:
        species = data.species.strip().title()
        personality = data.personality.strip().title()
        color = data.color.strip().title()

        s = encoders["Species"].transform([species])[0]
        p = encoders["Personality"].transform([personality])[0]
        c = encoders["Color 1"].transform([color])[0]

        df = pd.DataFrame([[s, p, c]],
                          columns=["Species_Num", "Personality_Num", "Color 1_Num"])

        probs = gift_model.predict_proba(df)[0]
        idx = probs.argmax()
        confidence = float(probs[idx])

        style_id = gift_model.classes_[idx]
        style = encoders["Style 1"].inverse_transform([style_id])[0]

        if confidence < 0.4:
            message = "🦉 'Hoo… how curious! This is a rather uncertain case, but this gift *may* pique their interest.' 🤔"
        elif confidence < 0.7:
            message = "🦉 'Ahh, fascinating! This gift should suit them quite nicely, I daresay!' 📜"
        elif confidence < 0.9:
            message = "🦉 'Hoo hoo! A splendid match! This gift aligns beautifully with their tastes!' 🎁"
        else:
            message = "🦉 'Extraordinary!! A perfect specimen of gifting! They shall be absolutely delighted!' 🌟"

        return {
            "recommended_gift": style,
            "confidence": round(confidence, 3),
            "message": message
        }

    except ValueError:
        return {
            "error": "🦉 'Hoo! I’m afraid I couldn’t recognize one of those traits. Do try again with known categories!'"
        }

    except Exception:
        return {"error": "Invalid input values"}

@app.post("/fishing-decision")
def fishing_decision(data: FishingDecisionInput):
    try:
        shadow = data.shadow.strip().title()
        location = data.location.strip().title()

        s_id = shadow_map.get(shadow, 0)
        l_id = encoders["Where/How"].transform([location])[0]

        df = pd.DataFrame([[s_id, l_id, data.spawn_rate]],
                          columns=["Shadow_Num", "Where_Num", "Spawn Rates"])

        pred = fishing_decision_model.predict(df)[0]

        probs = fishing_decision_model.predict_proba(df)[0]
        low_prob, high_prob = float(probs[0]), float(probs[1])

        # 🎯 Use HIGH VALUE probability directly
        if high_prob > 0.85:
            message = "🌟 'Ahh… the stars scream VALUE! This one's a jackpot!' 💫"
        elif high_prob > 0.6:
            message = "✨ 'I sense strong potential… this catch could be valuable!'"
        elif high_prob > 0.4:
            message = "🌙 'Hmm… uncertain tides. Could go either way.'"
        else:
            message = "🐟 'Bah! The cosmos says this one's hardly worth it.'"

        result = "💰 High Value Catch" if pred == 1 else "🐟 Common Catch"

        return {
            "is_high_value": int(pred),
            "result": result,
            "high_value_probability": round(high_prob, 3),
            "low_value_probability": round(low_prob, 3),
            "message": message
        }

    except ValueError:
        return {
            "error": "🌟 'Oh dear… I couldn’t read the stars from that input. Could you try again?'"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)