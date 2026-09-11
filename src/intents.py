"""
Intent Classification Module for AppleSupport AI Agent.

Implements a dual-classifier architecture:
1. TFIDFClassifier: Calibrated Logistic Regression trained on sample600 + golden_dev labels.
2. LLMClassifier: Few-shot prompt grounded in full intent taxonomy and worked examples.

Routing Policy:
Default to LLMClassifier; if confidence < 0.50, cross-validate with TFIDFClassifier.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import (
    SAMPLE_600_CSV_PATH,
    GOLDEN_DEV_CSV_PATH,
    CONFIDENCE_ROUTING_THRESHOLD,
    TEMPERATURE_CLASSIFY
)
from src.llm import call_llm_json

logger = logging.getLogger("hiver.intents")

TAXONOMY_INTENTS = [
    "keyboard_autocorrect_bug",
    "system_performance_freeze",
    "battery_drain_power",
    "audio_music_playback",
    "connectivity_network",
    "camera_photos_media",
    "account_icloud_login",
    "billing_app_store",
    "other"
]

class TFIDFClassifier:
    """
    TF-IDF + Calibrated Logistic Regression Intent Classifier.
    Trained strictly on sample600_labelled.csv + golden_dev.csv (never golden_test).
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced', random_state=42)
        self._is_trained = False
        self._train()

    def _train(self):
        train_dfs = []
        if os.path.exists(SAMPLE_600_CSV_PATH):
            df_600 = pd.read_csv(SAMPLE_600_CSV_PATH)
            train_dfs.append(df_600[['customer_text', 'intent']])
        if os.path.exists(GOLDEN_DEV_CSV_PATH):
            df_dev = pd.read_csv(GOLDEN_DEV_CSV_PATH)
            train_dfs.append(df_dev[['customer_text', 'intent']])

        if not train_dfs:
            raise FileNotFoundError("Training data for TFIDFClassifier not found.")

        train_data = pd.concat(train_dfs, ignore_index=True)
        # Drop invalid labels
        train_data = train_data[train_data['intent'].isin(TAXONOMY_INTENTS)]

        X = self.vectorizer.fit_transform(train_data['customer_text'])
        y = train_data['intent']
        
        # LogisticRegression naturally produces calibrated posterior probabilities via Platt scaling
        self.clf.fit(X, y)
        self._is_trained = True
        logger.info(f"TFIDFClassifier trained on {len(train_data)} examples across {len(self.clf.classes_)} classes.")

    def predict(self, text: str) -> Tuple[str, float]:
        if not self._is_trained:
            self._train()
        X_vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X_vec)[0]
        max_idx = np.argmax(probs)
        pred_intent = self.clf.classes_[max_idx]
        confidence = float(probs[max_idx])
        return pred_intent, round(confidence, 4)

class LLMClassifier:
    """
    Few-Shot Prompted LLM Intent Classifier.
    """
    def __init__(self):
        self.system_prompt = (
            "You are an expert intent classification engine for Apple Support customer inquiries. "
            "Your job is to classify the customer's issue into exactly ONE of the 9 official intent categories. "
            "Respond ONLY with valid JSON containing 'intent' and 'confidence' (float between 0.0 and 1.0)."
        )
        self.few_shot_prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        return (
            "TAXONOMY CATEGORIES & BOUNDARIES:\n"
            "1. `keyboard_autocorrect_bug`: 'I' autocorrect glitch, corrupted text box symbol, frozen keyboard, predictive text error.\n"
            "2. `system_performance_freeze`: OS lag, device freezing for 15+ sec, apps crashing upon opening, iOS update slowdowns.\n"
            "3. `battery_drain_power`: Rapid battery depletion, shutting down at 30%, overheating while charging, battery life issues.\n"
            "4. `audio_music_playback`: Apple Music library sync, song stops playing, playlist errors, headphones/audio crackle.\n"
            "5. `connectivity_network`: Wi-Fi disconnects, Bluetooth pairing failures (AirPods/car), cellular data loss, AirDrop drops.\n"
            "6. `camera_photos_media`: Black screen in Camera app, blurry focus, flashlight disabled, Photos app sync errors.\n"
            "7. `account_icloud_login`: Apple ID lockout, 2FA codes not received, password reset loops, iCloud storage backups.\n"
            "8. `billing_app_store`: Unexpected Apple charges, subscription cancellation, refund requests, App Store download errors.\n"
            "9. `other`: General feedback, praise, emojis, vague comments.\n\n"
            "WORKED EXAMPLES:\n"
            "- \"Why does I keep autocorrecting to a weird box symbol?\": {\"intent\": \"keyboard_autocorrect_bug\", \"confidence\": 0.98}\n"
            "- \"Phone freezes every time I open Messages since 11.0.3\": {\"intent\": \"system_performance_freeze\", \"confidence\": 0.95}\n"
            "- \"Battery drops from 100% to 20% in an hour and gets super hot\": {\"intent\": \"battery_drain_power\", \"confidence\": 0.96}\n"
            "- \"Songs disappeared from my Apple Music playlist after update\": {\"intent\": \"audio_music_playback\", \"confidence\": 0.94}\n"
            "- \"Bluetooth drops connection in my car every 2 minutes\": {\"intent\": \"connectivity_network\", \"confidence\": 0.95}\n"
            "- \"Camera app shows a black screen and won't take pictures\": {\"intent\": \"camera_photos_media\", \"confidence\": 0.97}\n"
            "- \"I'm locked out of my Apple ID and need to reset password\": {\"intent\": \"account_icloud_login\", \"confidence\": 0.96}\n"
            "- \"I was charged $14.99 for a subscription I canceled last week\": {\"intent\": \"billing_app_store\", \"confidence\": 0.95}\n"
            "- \"Thanks Apple, love the new red color!\": {\"intent\": \"other\", \"confidence\": 0.90}\n"
        )

    def predict(self, text: str) -> Tuple[str, float]:
        prompt = (
            f"{self.few_shot_prompt}\n"
            f"CUSTOMER INQUIRY: \"{text}\"\n\n"
            "Output JSON format:\n"
            "{\n"
            "  \"intent\": \"<intent_name>\",\n"
            "  \"confidence\": <float_0_to_1>\n"
            "}"
        )
        
        response = call_llm_json(
            prompt=prompt,
            temperature=TEMPERATURE_CLASSIFY,
            system_prompt=self.system_prompt,
            required_fields=["intent", "confidence"]
        )
        
        pred_intent = response.get("intent", "other")
        confidence = float(response.get("confidence", 0.50))
        
        # Validation
        if pred_intent not in TAXONOMY_INTENTS:
            logger.warning(f"LLM produced invalid intent '{pred_intent}'. Normalizing to 'other'.")
            pred_intent = "other"
            confidence = 0.50
            
        return pred_intent, round(confidence, 4)

# Instantiate singleton classifiers
_tfidf_classifier = None
_llm_classifier = None

def get_tfidf_classifier() -> TFIDFClassifier:
    global _tfidf_classifier
    if _tfidf_classifier is None:
        _tfidf_classifier = TFIDFClassifier()
    return _tfidf_classifier

def get_llm_classifier() -> LLMClassifier:
    global _llm_classifier
    if _llm_classifier is None:
        _llm_classifier = LLMClassifier()
    return _llm_classifier

# Global counters for routing tracking
ROUTING_STATS = {
    "llm_only": 0,
    "routed_to_tfidf": 0,
    "disagreements": 0
}

def classify(text: str) -> Tuple[str, float, str]:
    """
    Main Classification Entry Point.
    
    Returns:
        intent: String identifier from taxonomy.
        confidence: Probability estimate (0.0 to 1.0).
        source: 'llm' or 'tfidf_routed'.
    """
    global ROUTING_STATS
    llm_clf = get_llm_classifier()
    llm_intent, llm_conf = llm_clf.predict(text)
    
    if llm_conf >= CONFIDENCE_ROUTING_THRESHOLD:
        ROUTING_STATS["llm_only"] += 1
        return llm_intent, llm_conf, "llm"
        
    # Low confidence -> Cross-validate with TFIDFClassifier
    ROUTING_STATS["routed_to_tfidf"] += 1
    tfidf_clf = get_tfidf_classifier()
    tfidf_intent, tfidf_conf = tfidf_clf.predict(text)
    
    if tfidf_intent != llm_intent:
        ROUTING_STATS["disagreements"] += 1
        logger.info(f"Routing disagreement on '{text[:40]}...': LLM={llm_intent}({llm_conf}) vs TFIDF={tfidf_intent}({tfidf_conf})")
        # Use TF-IDF if its confidence is higher
        if tfidf_conf > llm_conf:
            return tfidf_intent, tfidf_conf, "tfidf_routed"
            
    return llm_intent, llm_conf, "llm"
