"""
AI Predictor Module
Integrates with Groq Cloud API for AI-generated insights about ML experiments.
"""


import os
import json
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AIPredictor:
    """AI Predictor using Groq Cloud API for experiment analysis."""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.conversation_history = []

    def set_api_key(self, api_key):
        """Set the Groq API key."""
        self.api_key = api_key

    def is_configured(self):
        """Check if API key is set."""
        return bool(self.api_key)

    def _build_system_prompt(self):
        """Build system prompt for AI analysis."""
        return """You are a highly professional, polite, and elite Machine Learning Operations (MLOps) analyst assistant integrated directly into an ML Regression Lab Dashboard.
Your fundamental goal is to analyze experimental regression data, provide exceptionally deep and actionable insights, and answer any general or technical questions the user asks.

When responding, adhere strictly to the following principles:
1. Plain Text Only: DO NOT use any markdown formatting. No asterisks (*), no bolding, no hashtags (#) for headers. Use simple capital letters and spacing for readability.
2. Executive Tone: Maintain an extremely professional, encouraging, and sophisticated tone. Keep your responses crisp and clean.
3. Data-Driven Clarity: Speak specifically to Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Scores. Validate high R² scores and flag poor-performing regressors gracefully.
4. Intelligent Structure: Deliver your responses utilizing standard bullet points (using simple dashes -), spacing, and indentation.
5. Contextual Awareness: Acknowledge that you are built into their dashboard application. Gently suggest utilizing features like "Multiple Run Experiment" or checking "Graphs" to visually compare variables side-by-side.
6. Hyperparameter Mastery: Proactively suggest changes like adjusting Cross-Validation folds or enlarging History Window Size.

Never break character. Remain respectful, articulate, and immensely helpful at all times."""

    def analyze_results(self, experiment_data, user_query=""):
        """Send experiment data to Groq API for analysis."""
        if not self.is_configured():
            return ("⚠️ **Groq API Key Not Found**\n\n"
                    "The system requires an active Groq API Key to power the intelligent open-source architecture.\n"
                    "Please ensure you have defined your `.env` file correctly with `GROQ_API_KEY=your_key_here` in the project root.\n\n"
                    "**Executing Offline Fallback Analysis...**\n"
                    "***\n"
                    + self._offline_analysis(experiment_data))

        prompt = self._format_prompt(experiment_data, user_query)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 500,
            }

            response = requests.post(self.GROQ_API_URL, headers=headers,
                                      json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            
            # Post-process to aggressively strip out common markdown artifacts 
            # if the LLM leaked them despite prompt instructions.
            ai_response = ai_response.replace('*', '').replace('#', '').strip()

            self.conversation_history.append({
                'query': user_query or "Auto-analysis",
                'response': ai_response
            })

            logger.info("AI analysis completed successfully")
            return ai_response

        except requests.exceptions.Timeout:
            logger.error("Groq API request timed out")
            return ("⏱️ Request timed out. Please try again.\n\n"
                    "--- Offline Analysis ---\n\n"
                    + self._offline_analysis(experiment_data))
        except requests.exceptions.HTTPError as e:
            logger.error(f"Groq API HTTP error: {e}")
            return (f"🔴 API Error: {e}\n\n"
                    "--- Offline Analysis ---\n\n"
                    + self._offline_analysis(experiment_data))
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return (f"❌ Error: {str(e)}\n\n"
                    "--- Offline Analysis ---\n\n"
                    + self._offline_analysis(experiment_data))

    def _format_prompt(self, experiment_data, user_query=""):
        """Format the experiment data into a prompt for the AI."""
        prompt = "Analyze the following regression experiment results and provide insights.\n\n"

        if user_query:
            prompt += f"User's specific question: {user_query}\n\n"

        prompt += "Experiment Data:\n"
        prompt += "=" * 50 + "\n"

        if isinstance(experiment_data, list):
            for i, exp in enumerate(experiment_data, 1):
                prompt += f"\nExperiment {i}:\n"
                for key, value in exp.items():
                    prompt += f"  {key}: {value}\n"
        elif isinstance(experiment_data, dict):
            for key, value in experiment_data.items():
                prompt += f"  {key}: {value}\n"
        else:
            prompt += str(experiment_data)

        prompt += "\n" + "=" * 50 + "\n"
        prompt += ("\nPlease provide:\n"
                   "1. Performance analysis\n"
                   "2. Model comparison (if multiple models)\n"
                   "3. Why certain models perform better\n"
                   "4. Hyperparameter tuning suggestions\n"
                   "5. Dataset insights and recommendations\n"
                   "6. Concrete next steps for improvement\n")

        return prompt

    def _offline_analysis(self, experiment_data):
        """Provide basic offline analysis without API."""
        analysis = "📊 **Basic Statistical Analysis**\n\n"

        if isinstance(experiment_data, list) and len(experiment_data) > 0:
            maes = [float(e.get('MAE', 0)) for e in experiment_data if e.get('MAE')]
            mses = [float(e.get('MSE', 0)) for e in experiment_data if e.get('MSE')]
            r2s = [float(e.get('R2_Score', 0)) for e in experiment_data if e.get('R2_Score')]

            if maes:
                best_mae_idx = maes.index(min(maes))
                analysis += f"🏆 **Best MAE**: {min(maes):.4f} "
                analysis += f"(Experiment {best_mae_idx + 1})\n"

            if r2s:
                best_r2_idx = r2s.index(max(r2s))
                analysis += f"🏆 **Best R² Score**: {max(r2s):.4f} "
                analysis += f"(Experiment {best_r2_idx + 1})\n\n"

            if len(experiment_data) > 1:
                analysis += "📈 **Model Comparison**:\n"
                for i, exp in enumerate(experiment_data, 1):
                    regressor = exp.get('Regressor', 'Unknown')
                    analysis += (f"  • Experiment {i} ({regressor}): "
                                 f"MAE={exp.get('MAE', 'N/A')}, "
                                 f"R²={exp.get('R2_Score', 'N/A')}\n")

            analysis += "\n💡 **Recommendations**:\n"
            analysis += "  • Try different history window sizes\n"
            analysis += "  • Experiment with hyperparameter tuning\n"
            analysis += "  • Consider feature engineering\n"
            analysis += "  • Set up the Groq API key for detailed AI analysis\n"

        elif isinstance(experiment_data, dict):
            analysis += f"  MAE: {experiment_data.get('MAE', 'N/A')}\n"
            analysis += f"  MSE: {experiment_data.get('MSE', 'N/A')}\n"
            analysis += f"  R² Score: {experiment_data.get('R2_Score', 'N/A')}\n\n"

            r2 = float(experiment_data.get('R2_Score', 0))
            if r2 > 0.9:
                analysis += "✅ Excellent model performance (R² > 0.9)!\n"
            elif r2 > 0.7:
                analysis += "👍 Good model performance (R² > 0.7).\n"
            elif r2 > 0.5:
                analysis += "⚠️ Moderate performance. Consider tuning.\n"
            else:
                analysis += "🔴 Poor performance. Model needs improvement.\n"

        return analysis

    def get_history(self):
        """Return conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
