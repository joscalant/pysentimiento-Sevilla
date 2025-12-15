# pysentimiento-Sevilla


import pandas as pd
from pysentimiento import create_analyzer

# -----------------------------
# 1. Load CSV
# -----------------------------
df = pd.read_csv('D:/JCA/Doctorado - US/Sevilla segregación/Paper Miguel/Metodologia de analisis cualitativo/Python/comentarios.csv')
  # Adjust filename if needed
TEXT_COLUMN = "comment"

# -----------------------------
# 2. Create analyzers (Spanish)
# -----------------------------
analyzer_sentiment = create_analyzer(task="sentiment", lang="es")
analyzer_emotion = create_analyzer(task="emotion", lang="es")
analyzer_hate = create_analyzer(task="hate_speech", lang="es")
analyzer_irony = create_analyzer(task="irony", lang="es")

# -----------------------------
# 3. Functions for each task
# -----------------------------
def analyze_sentiment(text):
    result = analyzer_sentiment.predict(text)
    return pd.Series({
        "sentiment_label": result.output,
        "sentiment_pos": result.probas.get("POS"),
        "sentiment_neg": result.probas.get("NEG"),
        "sentiment_neu": result.probas.get("NEU"),
    })

def analyze_emotion(text):
    result = analyzer_emotion.predict(text)
    return pd.Series({
        "emotion_label": result.output,
        **{f"emotion_{k}": v for k, v in result.probas.items()}
    })

def analyze_hate(text):
    result = analyzer_hate.predict(text)
    return pd.Series({
        "hate_label": result.output,
        **{f"hate_{k}": v for k, v in result.probas.items()}
    })

def analyze_irony(text):
    result = analyzer_irony.predict(text)
    return pd.Series({
        "irony_label": result.output,
        **{f"irony_{k}": v for k, v in result.probas.items()}
    })


# -----------------------------
# 4. Apply every model
# -----------------------------
df_results = df.copy()

df_results = df_results.join(df_results[TEXT_COLUMN].apply(analyze_sentiment))
df_results = df_results.join(df_results[TEXT_COLUMN].apply(analyze_emotion))
df_results = df_results.join(df_results[TEXT_COLUMN].apply(analyze_hate))
df_results = df_results.join(df_results[TEXT_COLUMN].apply(analyze_irony))


# -----------------------------
# 5. Save results
# -----------------------------
df_results.to_csv('D:/JCA/Doctorado - US/Sevilla segregación/Paper Miguel/Metodologia de analisis cualitativo/Python/analysis_results.csv', index=False)

