""" D1.3 - Learning systems, text, recommendations and speech

We will explore several kinds of AI systems using tiny examples. Each part shows the input, the operation and the result so learners can see what the system is doing.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Rank ideas by readiness

# We score value, available data and difficulty. This helps us choose an achievable first project.

print("\nStep 1 - Rank ideas by readiness")

ideas = pd.DataFrame({'idea': ['A', 'B'], 'value': [5, 3], 'data': [2, 5], 'feasibility': [3, 5]})
ideas['total'] = ideas['value'] + ideas['data'] + ideas['feasibility']
print(ideas.sort_values('total', ascending=False))

# %% 2. Mask personal information

# We replace an email before processing a text. This demonstrates data minimisation.

print("\nStep 2 - Mask personal information")

import re
text = 'Please reply to sam@example.com'
masked = re.sub(r'[\w.]+@[\w.]+', '[EMAIL]', text)
print(masked)

# %% 3. Words, stop words and pairs

# We split text into words, remove a common word and list adjacent word pairs.

print("\nStep 3 - Words, stop words and pairs")

words = 'the card is blocked'.split()
useful = [word for word in words if word != 'the']
print('Words:', words)
print('Without the:', useful)
print('Word pairs:', list(zip(useful, useful[1:])))

# %% 4. Counts and TF-IDF

# We turn words into numeric columns so a model can use text.

print("\nStep 4 - Counts and TF-IDF")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
texts = ['card fee', 'card problem', 'loan fee']
counts = CountVectorizer()
print('Words:', counts.fit(texts).get_feature_names_out())
print('Counts:\n', counts.transform(texts).toarray())
tfidf = TfidfVectorizer()
print('TF-IDF:\n', tfidf.fit_transform(texts).toarray().round(2))

# %% 5. Classify short texts

# We train a small classifier to show the input and output of complaint routing.

print("\nStep 5 - Classify short texts")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
texts = ['card blocked', 'card failed', 'loan payment', 'loan term']
labels = ['card', 'card', 'loan', 'loan']
model = make_pipeline(CountVectorizer(), LogisticRegression())
model.fit(texts, labels)
print('Predicted label:', model.predict(['card problem'])[0])

# %% 6. Compare with a baseline

# A baseline always chooses the most common label. We compare it with a prediction on a small check set.

print("\nStep 6 - Compare with a baseline")

from sklearn.metrics import accuracy_score
actual = ['card', 'loan', 'card', 'loan']
baseline = ['card', 'card', 'card', 'card']
example = ['card', 'loan', 'card', 'card']
print('Baseline accuracy:', accuracy_score(actual, baseline))
print('Example accuracy:', accuracy_score(actual, example))

# %% 7. Explain words, identify a number, and escalate

# We identify an amount and flag a word for human review. Real systems require wider testing.

print("\nStep 7 - Explain words, identify a number, and escalate")

import re
message = 'urgent fee 25'
print('Amount:', re.findall(r'\d+', message))
print('Human review:', 'urgent' in message)
print('Helpful words:', message.split())

# %% 8. Support, confidence and lift

# We count items bought together. Lift compares a conditional rate with the overall rate.

print("\nStep 8 - Support, confidence and lift")

a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
b = np.array([1, 1, 1, 0, 1, 0, 0, 0])
support = ((a == 1) & (b == 1)).mean()
confidence = b[a == 1].mean()
lift = confidence / b.mean()
print('Support:', support, 'Confidence:', confidence, 'Lift:', lift)

# %% 9. Recommend from similar choices

# We use a shared selection to suggest an item another person chose.

print("\nStep 9 - Recommend from similar choices")

choices = {'person 1': {'A', 'B'}, 'person 2': {'A', 'C'}}
print('Shared:', choices['person 1'] & choices['person 2'])
print('Suggested to person 1:', choices['person 2'] - choices['person 1'])

# %% 10. Cold start and coverage

# A new person has no history. We can begin with popular choices and count how many different items we show.

print("\nStep 10 - Cold start and coverage")

suggestions = ['A', 'B', 'A', 'C']
print('Fallback for new person: A')
print('Different items shown:', len(set(suggestions)))

# %% 11. Sampling and frequency

# Sound is represented by samples. The Nyquist limit is half the sampling rate.

print("\nStep 11 - Sampling and frequency")

sample_rate = 8000
print('Samples per second:', sample_rate)
print('Highest represented frequency:', sample_rate / 2, 'Hz')

# %% 12. Waveform and frequencies

# We construct a simple wave and look at the strongest frequency.

print("\nStep 12 - Waveform and frequencies")

rate = 100
time = np.arange(100) / rate
wave = np.sin(2 * np.pi * 5 * time)
frequencies = np.fft.rfftfreq(len(wave), 1 / rate)
strength = np.abs(np.fft.rfft(wave))
print('Strongest frequency:', frequencies[strength.argmax()], 'Hz')

# %% 13. Spectrogram and telephone keypad tones

# A spectrogram shows how frequencies change over time. Keypad tones use two frequencies together.

print("\nStep 13 - Spectrogram and telephone keypad tones")

from scipy.signal import spectrogram
rate = 8000
time = np.arange(800) / rate
tone = np.sin(2*np.pi*697*time) + np.sin(2*np.pi*1209*time)
frequencies, times, power = spectrogram(tone, fs=rate, nperseg=256)
print('Frequency bins:', len(frequencies), 'Time windows:', len(times))
strong = np.argsort(power.mean(axis=1))[-2:]
print('Strongest frequencies (approximately):', frequencies[strong])

# %% 14. Speech features and noise

# We compare the average magnitude of a simple wave before and after adding noise. Speech systems use richer features such as MFCCs.

print("\nStep 14 - Speech features and noise")

rng = np.random.default_rng(0)
wave = np.sin(np.linspace(0, 6, 100))
noisy = wave + rng.normal(0, 0.5, 100)
print('Clean average magnitude:', round(np.abs(wave).mean(), 2))
print('Noisy average magnitude:', round(np.abs(noisy).mean(), 2))

# %% 15. Recommendation hit rate at two

# We check whether the actual choice appears among the first two suggestions.

print("\nStep 15 - Recommendation hit rate at two")

suggested = ['A', 'B']
actual = 'B'
print('Hit in top two:', actual in suggested)
print('Precision at two:', int(actual in suggested) / len(suggested))

# %% 16. Word error rate and downstream errors

# We count one substituted word. An incorrect transcript can also change a later text classification.

print("\nStep 16 - Word error rate and downstream errors")

reference = 'show my balance'
transcript = 'show my challenge'
substitutions = 1
wer = substitutions / len(reference.split())
print('Reference:', reference)
print('Transcript:', transcript)
print('Word error rate:', round(wer, 2))
