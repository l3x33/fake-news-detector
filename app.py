
# Instruction: 
# 1. pip install flask joblib scikit-learn nltk trafilatura requests
# 2. python app.py
# -> http://127.0.0.1:5000

import re
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import trafilatura
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)


MODELS_DIR = Path(__file__).parent / 'models'
vectorizer  = joblib.load(MODELS_DIR / 'vectorizer.joblib')
final_model = joblib.load(MODELS_DIR / 'lr_final.joblib')
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients  = final_model.coef_[0]

STOP_WORDS = set(stopwords.words('english'))
stemmer    = PorterStemmer()


MESSAGES = {
    'fr': {
        'no_input':    "Aucun texte ou URL fourni.",
        'fetch_fail':  "Impossible de récupérer cette page (site bloqué, paywall, ou URL "
                    "invalide). Essayez de copier-coller le texte de l'article directement.",
        'extract_fail': "Le contenu extrait est trop court ou vide -- le site utilise peut-être "
                    "un format non standard. Essayez de copier-coller le texte directement.",
        'too_short':   "Texte trop court après nettoyage pour une prédiction fiable "
                    "(minimum recommandé : une vingtaine de mots significatifs).",
        'unexpected':  "Erreur inattendue : {e}",
    },
    'en': {
        'no_input':    "No text or URL provided.",
        'fetch_fail':  "Could not retrieve this page (blocked site, paywall, or invalid "
                    "URL). Try pasting the article text directly instead.",
        'extract_fail': "The extracted content is too short or empty -- the site may use a "
                        "non-standard format. Try pasting the text directly instead.",
        'too_short':   "Text too short after cleaning for a reliable prediction "
                    "(recommended minimum: about twenty meaningful words).",
        'unexpected':  "Unexpected error: {e}",
    },
}

def msg(lang, key, **kwargs):
    text = MESSAGES.get(lang, MESSAGES['fr'])[key]
    return text.format(**kwargs) if kwargs else text


def clean_full(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+|@\w+|#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(w for w in (stemmer.stem(w) for w in text.split())
                    if w not in STOP_WORDS and len(w) > 2)


def extract_article_text(url, lang='fr'):
    """Télécharge et extrait le texte principal d'un article via trafilatura
    (supprime navigation, pubs, footers -- garde uniquement le corps éditorial)."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError(msg(lang, 'fetch_fail'))
    text = trafilatura.extract(downloaded, favor_recall=True)
    if not text or len(text.strip()) < 50:
        raise ValueError(msg(lang, 'extract_fail'))
    return text


def predict(text, lang='fr'):
    """Pipeline complet : nettoyage -> vectorisation -> prédiction.
    Retourne la probabilité, la classe, et les mots les plus contributifs
    présents dans le texte (pour l'explication / le panneau 'indices')."""
    cleaned = clean_full(text)
    if len(cleaned.split()) < 5:
        raise ValueError(msg(lang, 'too_short'))

    vec  = vectorizer.transform([cleaned])
    prob_real = float(final_model.predict_proba(vec)[0, 1])
    pred_label = 'VRAI' if prob_real >= 0.5 else 'FAUX'

    present_idx = vec.nonzero()[1]
    contributions = []
    for idx in present_idx:
        word = feature_names[idx]
        coef = float(coefficients[idx])
        tfidf_val = float(vec[0, idx])
        contributions.append({
            'word': word,
            'coef': coef,
            'weight': coef * tfidf_val,
        })
    contributions.sort(key=lambda c: c['weight'])
    toward_fake = [c for c in contributions if c['weight'] < 0][:8]
    toward_real = [c for c in contributions if c['weight'] > 0][-8:][::-1]

    return {
        'prob_real': prob_real,
        'prob_fake': 1 - prob_real,
        'pred_label': pred_label,
        'n_words_analyzed': len(cleaned.split()),
        'n_words_raw': len(text.split()),
        'toward_fake': toward_fake,
        'toward_real': toward_real,
    }


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True) or {}
    mode = data.get('mode', 'text')
    lang = data.get('lang', 'fr')
    if lang not in MESSAGES:
        lang = 'fr'
    raw_input = (data.get('input') or '').strip()

    if not raw_input:
        return jsonify({'error': msg(lang, 'no_input')}), 400

    try:
        if mode == 'url':
            article_text = extract_article_text(raw_input, lang=lang)
        else:
            article_text = raw_input

        result = predict(article_text, lang=lang)
        result['source_mode'] = mode
        result['excerpt'] = article_text[:600]
        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': msg(lang, 'unexpected', e=e)}), 500


if __name__ == '__main__':
    print(f"Modèle chargé : {type(final_model).__name__}")
    print(f"Vocabulaire   : {len(feature_names):,} mots")
    print("Serveur démarré sur http://127.0.0.1:5000")
    app.run(debug=True, port=5000)