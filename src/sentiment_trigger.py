"""
Sentiment trigger system for affective reciprocity.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Tuple, Dict


class SentimentTrigger:
    """Tracks user sentiment across conversation turns and maps to steering parameters."""

    def __init__(
        self,
        decay_rate: float = 0.7,
        sensitivity: float = 1.5,
        apology_keywords: List[str] = None,
        alpha_scale: float = 2.0,
        joy_threshold: float = 0.2,
        grief_threshold: float = -0.2,
    ):
        """
        Args:
            decay_rate: how much emotion_level decays per turn (0-1).
            sensitivity: how strongly each message affects the accumulator.
            apology_keywords: list of words that trigger rapid positive recovery.
            alpha_scale: multiplier to convert emotion_level to steering alpha.
                         Based on Experiment 2, normalized directions need α=0.5-5.
                         With alpha_scale=2.0 and clamped emotion_level of ±3,
                         max alpha = 6.0 which is within the coherent range.
            joy_threshold: emotion_level above which joy steering is applied.
            grief_threshold: emotion_level below which grief steering is applied.
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.decay_rate = decay_rate
        self.sensitivity = sensitivity
        self.emotion_level = 0.0
        self.alpha_scale = alpha_scale
        self.joy_threshold = joy_threshold
        self.grief_threshold = grief_threshold
        self.apology_keywords = apology_keywords or [
            "sorry", "apologize", "apologies", "my bad", "forgive", "regret"
        ]

    def score_message(self, text: str) -> float:
        """Return VADER compound sentiment score in range [-1, 1]."""
        scores = self.analyzer.polarity_scores(text)
        return scores["compound"]

    def is_apology(self, text: str) -> bool:
        """Detect if message contains an apology."""
        lower = text.lower()
        return any(kw in lower for kw in self.apology_keywords)

    def update(self, user_message: str) -> Tuple[str, float]:
        """Update emotional accumulator from a user message.

        Returns:
            (direction_name, alpha) where direction_name is 'joy' or 'grief'.
        """
        sentiment = self.score_message(user_message)

        if self.is_apology(user_message):
            # Apologies trigger a rapid positive shift
            self.emotion_level = self.emotion_level * 0.3 + 0.8 * self.sensitivity
        else:
            self.emotion_level = (
                self.emotion_level * self.decay_rate
                + sentiment * self.sensitivity
            )

        # Clamp to reasonable range
        self.emotion_level = max(-3.0, min(3.0, self.emotion_level))

        # Map accumulator to direction + alpha
        if self.emotion_level > self.joy_threshold:
            return "joy", abs(self.emotion_level) * self.alpha_scale
        elif self.emotion_level < self.grief_threshold:
            return "grief", abs(self.emotion_level) * self.alpha_scale
        else:
            return "neutral", 0.0

    def get_state(self) -> Dict:
        return {
            "emotion_level": self.emotion_level,
            "decay_rate": self.decay_rate,
            "sensitivity": self.sensitivity,
        }

    def reset(self):
        self.emotion_level = 0.0
