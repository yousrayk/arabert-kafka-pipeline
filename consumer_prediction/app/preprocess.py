"""AraBERT text preprocessing — the single definition shared by serving
(this service) and training (training/train.py imports this module), so the
model never sees text cleaned differently from what it was trained on.
"""

MODEL_NAME = "aubmindlab/bert-base-arabertv02"


class ArabicPreprocessor:
    """Thin wrapper over the official arabert ArabertPreprocessor, configured
    for the v02 model: strips tashkeel/tatweel, normalizes repeated
    characters, and replaces URLs/mentions/emails with the special tokens
    the model was pre-trained with. v02 needs no Farasa segmentation, so
    no Java is required."""

    def __init__(self, model_name: str = MODEL_NAME):
        from arabert.preprocess import ArabertPreprocessor

        self._preprocessor = ArabertPreprocessor(model_name=model_name)

    def __call__(self, text: str) -> str:
        return self._preprocessor.preprocess(text).strip()
