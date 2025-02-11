# Data

This section provides an overview of the data used in this project. The data is crucial for understanding the context and the results of the analysis.

## Data Sources

- **RAVDESS**: Speech audio-only files (16bit, 48kHz .wav) from the RAVDESS.

## Data Structure

**RAVDESS**: Here is the filename identifiers as per the official RAVDESS website:

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

_Filename example: 03-01-06-01-02-01-12.wav_

1. Audio-only (03)
2. Speech (01)
3. Fearful (06)
4. Normal intensity (01)
5. Statement "dogs" (02)
6. 1st Repetition (01)
7. 12th Actor (12)

Female, as the actor ID number is even.

## Additional Notes

- **RAVDESS** directory structure:

```
RAVDESS/
├── Speech/
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ...
└── Singing/
    ├── Actor_01/
    ├── Actor_02/
    └── ...
```

## Citation

Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.
