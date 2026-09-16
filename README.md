# AI Spam Classifier (SMS / Email)

TF-IDF + Multinomial Naive Bayes spam classifier built with Python & Scikit-learn.

## Files
- `data/sms_data.csv` — labeled dataset (label, text)
- `generate_data.py` — (re)generates the sample dataset
- `train.py` — trains the model and saves it to `model/`
- `predict.py` — classifies new messages using the saved model
- `model/` — saved `spam_model.pkl` and `vectorizer.pkl`

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py
```

## Predict
```bash
python predict.py "You won a free prize, click here now!"
python predict.py        # interactive mode
```

## Use your own dataset
Replace `data/sms_data.csv` with any CSV containing `label` (spam/ham) and
`text` columns — e.g. the UCI SMS Spam Collection — then rerun `train.py`.

## Notes
- Swap `MultinomialNB` in `train.py` for `LogisticRegression` or `LinearSVC`
  for potentially higher accuracy on larger real-world datasets.
- Increase `max_features` in `TfidfVectorizer` for larger datasets.
